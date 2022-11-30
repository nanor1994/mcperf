from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
import argparse
import functools
import logging
import os
import sys
import socket
import time
import threading
import subprocess
from subprocess import call
import re
import math

# TODO: ProfilerGroup has a tick thread that wakes up at the minimum sampling period and wakes up each profiler if it has to wake up
# TODO: Use sampling period and sampling length
# def power_state_diff(new_vector, old_vector):
#     diff = []
#     for (new, old) in zip(new_vector, old_vector):
#         diff.append([x[0] - x[1] for x in zip(new, old)])
#     return diff

class EventProfiling:
    def __init__(self, sampling_period = 0, sampling_length = 1):
        self.terminate_thread = threading.Condition()
        self.is_active = False
        self.sampling_period = sampling_period
        self.sampling_length = sampling_length

    def profile_thread(self):
        logging.info("Profiling thread started")
        self.terminate_thread.acquire()
        while self.is_active:
            timestamp = str(int(time.time()))
            self.terminate_thread.release()
            self.sample(timestamp)
            self.terminate_thread.acquire()
            if self.is_active:
                self.terminate_thread.wait(timeout=self.sampling_period - self.sampling_length)
        self.terminate_thread.release()
        timestamp = str(int(time.time()))
        self.zerosample(timestamp)
        logging.info("Profiling thread terminated")

    def start(self):

        self.clear()
        if self.sampling_period:

            self.is_active=True
            self.thread = threading.Thread(target=EventProfiling.profile_thread, args=(self,))
            self.thread.daemon = True
            self.thread.start()
        else:

            timestamp = str(int(time.time()))
            self.sample(timestamp)

    def stop(self):
        if self.sampling_period:
            self.terminate_thread.acquire()
            self.interrupt_sample()
            self.is_active=False
            self.terminate_thread.notify()
            self.terminate_thread.release()
        else:
            timestamp = str(int(time.time()))
            self.sample(timestamp)


class PerfEventProfiling(EventProfiling):
    def __init__(self, sampling_period=1, sampling_length=1, iteration=1):
        print(sampling_period)
        super().__init__(sampling_period, sampling_length)
        self.perf_path = self.find_perf_path()

        logging.info('Perf found at {}'.format(self.perf_path))

        self.events = PerfEventProfiling.get_microarchitectural_events()
        self.perf_stats_events = PerfEventProfiling.get_perf_stat_events()

        self.timeseries = {}
        self.iteration=iteration

        for e in self.perf_stats_events:
            self.timeseries[e] = []

        #get pid of memcached    
        cmd = ['pgrep', 'memcached']
        result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out = result.stdout.decode('utf-8').splitlines() + result.stderr.decode('utf-8').splitlines()
        self.pid=out[0]
        print(str(self.pid))

    def find_perf_path(self):
        kernel_uname = os.popen('uname -a').read().strip()
        if '4.15.0-159-generic' in kernel_uname:
            return 'perf'
        else:
            return 'perf'

    def get_perf_power_events(self):
        events = []
        result = subprocess.run([self.perf_path, 'list'], stdout=subprocess.PIPE)
        for l in result.stdout.decode('utf-8').splitlines():
            l = l.lstrip()
            m = re.match("(power/energy-.*/)\s*\[Kernel PMU event]", l)
            if m:
                events.append(m.group(1))

    @staticmethod
    def get_microarchitectural_events():
        events = []

        events.append("instructions:u")
        events.append("cycles:u")
        events.append("instructions:k")
        events.append("cycles:k")
        events.append("BR_MISP_RETIRED.ALL_BRANCHES")
        events.append("L1-icache-load-misses")
        events.append("L1-dcache-load-misses")
        events.append("dTLB-load-misses")
        events.append("iTLB-load-misses")
        events.append("mem_load_retired.l2_miss")
        events.append("mem_load_retired.l3_miss")
        return events

    @staticmethod
    def get_perf_stat_events():
        ev=[]

        ev.append("instructions:u")
        ev.append("cycles:u")
        ev.append("instructions:k")
        ev.append("cycles:k")
        ev.append("BR_MISP_RETIRED.ALL_BRANCHES")
        ev.append("L1-icache-load-misses")
        ev.append("L1-dcache-load-misses")
        ev.append("dTLB-load-misses")
        ev.append("iTLB-load-misses")
        ev.append("mem_load_retired.l2_miss")
        ev.append("mem_load_retired.l3_miss")
        
        
        return ev

    def sample(self, timestamp):

        iterations_cycle=math.ceil((len(self.events))/4.0)  #4 number of available perf counters provided by intel +1 in orer to run perf stat without events
        event_index=self.iteration%iterations_cycle
        event_index=event_index*4

        if (event_index+4) >= len(self.events) :
            events_str = ','.join(self.events[(event_index):])
        else:
            events_str = ','.join(self.events[(event_index):(event_index+4)])

        if events_str=="":
            cmd = ['sudo', self.perf_path, 'stat', '-a','sleep', str(self.sampling_length)]
        else:
            cmd = ['sudo', self.perf_path, 'stat', '-e', events_str, '-p', self.pid, 'sleep', str(self.sampling_length)]
        # -p self.pid
        result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out = result.stdout.decode('utf-8').splitlines() + result.stderr.decode('utf-8').splitlines()
        print(out)

        for e in self.perf_stats_events:

            for l in out:
                l = l.lstrip()
                m = re.match("(.*)\s+.*\s+{}".format(e), l)

                if m:
                    value = m.group(1)
                    self.timeseries[e].append((timestamp, str(float(value.replace(',', '')))))


    # FIXME: Currently, we add a dummy zero sample when we finish sampling.
    # This helps us to determine the sampling duration later when we analyze the stats
    # It would be nice to have a more clear solution
    def zerosample(self, timestamp):
        for e in self.perf_stats_events:
            self.timeseries[e].append((timestamp, str(0.0)))

    def interrupt_sample(self):
        os.system('sudo pkill -2 sleep')

    def clear(self):
        self.timeseries = {}
        for e in self.perf_stats_events:
            self.timeseries[e] = []

    def report(self):
        return self.timeseries

class StateProfiling(EventProfiling):
    cpuidle_path = '/sys/devices/system/cpu/cpu0/cpuidle/'

    def __init__(self, sampling_period=0):
        super().__init__(sampling_period)
        self.state_names = StateProfiling.power_state_names()
        self.timeseries = {}

    @staticmethod
    def power_state_names():
        cpuidle_path = StateProfiling.cpuidle_path
        if not os.path.exists(cpuidle_path):
            return []
        state_names = []
        states = os.listdir(cpuidle_path)
        states.sort()
        for state in states:
            state_name_path = os.path.join(cpuidle_path, state, 'name')
            with open(state_name_path) as f:
                state_names.append(f.read().strip())
        return state_names

    @staticmethod
    def power_state_metric(cpu_id, state_id, metric):
        cpuidle_path = StateProfiling.cpuidle_path
        if not os.path.exists(cpuidle_path):
            return None
        output = open("/sys/devices/system/cpu/cpu{}/cpuidle/state{}/{}".format(cpu_id, state_id, metric)).read()
        return output.strip()

    def sample_power_state_metric(self, metric, timestamp):
        for cpu_id in range(0, os.cpu_count()):
            for state_id in range(0, len(self.state_names)):
                state_name = self.state_names[state_id]
                key = "CPU{}.{}.{}".format(cpu_id, state_name, metric)
                value = StateProfiling.power_state_metric(cpu_id, state_id, metric)
                self.timeseries.setdefault(key, []).append((timestamp, value))

    def sample(self, timestamp):
        self.sample_power_state_metric('usage', timestamp)
        self.sample_power_state_metric('time', timestamp)

    def interrupt_sample(self):
        pass

    def zerosample(self, timestamp):
        pass

    def clear(self):
        self.timeseries = {}

    def report(self):
        return self.timeseries

class RaplCountersProfiling(EventProfiling):
    raplcounters_path = '/sys/class/powercap/intel-rapl/'

    def __init__(self, sampling_period=0):
        super().__init__(sampling_period)
        self.domain_names = {}
        self.domain_names = RaplCountersProfiling.power_domain_names()
        self.timeseries = {}

    @staticmethod
    def power_domain_names():
        raplcounters_path = RaplCountersProfiling.raplcounters_path
        if not os.path.exists(raplcounters_path):
            return []
        domain_names = {}

        #Find all supported domains of the system
        for root, subdirs, files in os.walk(raplcounters_path):
            for subdir in subdirs:
                if "intel-rapl" in subdir:
                    domain_names[open("{}/{}/{}".format(root, subdir,'name'), "r").read().strip()]= os.path.join(root,subdir,'energy_uj')
        return domain_names


    def sample(self, timestamp):
         for domain in self.domain_names:
                value = open(self.domain_names[domain], "r").read().strip()
                self.timeseries.setdefault(domain, []).append((timestamp, value))


    def interrupt_sample(self):
        pass

    def zerosample(self, timestamp):
        pass

    def clear(self):
        self.timeseries = {}

    def report(self):
        return self.timeseries

class ProfilingService:
    def __init__(self, profilers):
        self.profilers = profilers

    def start(self):

        for p in self.profilers:
            print(p)
            p.start()

    def stop(self):
        for p in self.profilers:
            p.stop()
        time.sleep(5)

    def report(self):
        timeseries = {}
        time.sleep(5)
        for p in self.profilers:
            t = p.report()
            timeseries = {**timeseries, **t}
        return timeseries

    def set(self, kv):
        print(kv)


def server(port,perf_iteration):
    perf_event_profiling = PerfEventProfiling(sampling_period=120,sampling_length=180,iteration=perf_iteration)
    state_profiling = StateProfiling(sampling_period=0)
    rapl_profiling = RaplCountersProfiling(sampling_period=0)

    profiling_service = ProfilingService([rapl_profiling, state_profiling, perf_event_profiling])

    hostname = socket.gethostname().split('.')[0]
    server = SimpleXMLRPCServer((hostname, port), allow_none=True)
    server.register_instance(profiling_service)
    logging.info("Listening on port {}...".format(port))
    server.serve_forever()

class StartAction:
    @staticmethod
    def add_parser(subparsers):
        parser = subparsers.add_parser('start', help = "Start profiling")
        parser.set_defaults(func=StartAction.action)

    @staticmethod
    def action(args):
        with xmlrpc.client.ServerProxy("http://{}:{}/".format(args.hostname, args.port)) as proxy:
            proxy.start()

class StopAction:
    @staticmethod
    def add_parser(subparsers):
        parser = subparsers.add_parser('stop', help = "Stop profiling")
        parser.set_defaults(func=StopAction.action)

    @staticmethod
    def action(args):
        with xmlrpc.client.ServerProxy("http://{}:{}/".format(args.hostname, args.port)) as proxy:
            proxy.stop()

class ReportAction:
    @staticmethod
    def add_parser(subparsers):
        parser = subparsers.add_parser('report', help = "Report profiling")
        parser.set_defaults(func=ReportAction.action)
        parser.add_argument(
                    "-d", "--directory", dest='directory',
                    help="directory where to output results")

    @staticmethod
    def action(args):
        with xmlrpc.client.ServerProxy("http://{}:{}/".format(args.hostname, args.port)) as proxy:
            stats = proxy.report()
            if args.directory:
                ReportAction.write_output(stats, args.directory)
            else:
                print(stats)

    @staticmethod
    def write_output(stats, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for metric_name,timeseries in stats.items():
            metric_file_name = metric_name.replace('/', '-')
            metric_file_path = os.path.join(directory, metric_file_name)
            with open(metric_file_path, 'w') as mf:
                mf.write(metric_name + '\n')
                for val in timeseries:
                    mf.write(','.join(val) + '\n')

class SetAction:
    @staticmethod
    def add_parser(subparsers):
        parser = subparsers.add_parser('set', help = "Set sysfs")
        parser.set_defaults(func=SetAction.action)
        parser.add_argument('-c', dest='command')
        parser.add_argument('rest', nargs=argparse.REMAINDER)

    @staticmethod
    def action(args):
        with xmlrpc.client.ServerProxy("http://{}:{}/".format(args.hostname, args.port)) as proxy:
            proxy.set(args.rest)

def parse_args():
    """Configures and parses command-line arguments"""
    parser = argparse.ArgumentParser(
                    prog = 'profiler',
                    description='profiler',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-n", "--hostname", dest='hostname',
        help="profiler server hostname")
    parser.add_argument(
        "-p", "--port", dest='port', type=int, default=8000,
        help="profiler server port")
    parser.add_argument(
        "-v", "--verbose", dest='verbose', action='store_true',
        help="verbose")

    parser.add_argument(
        "-i", "--iteration", dest='perf_iteration', type=int, default=1,
        help="perf iteration to choose the right performance counters")

    subparsers = parser.add_subparsers(dest='subparser_name', help='sub-command help')
    actions = [StartAction, StopAction, ReportAction, SetAction]
    for a in actions:
      a.add_parser(subparsers)

    args = parser.parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s')

    if args.verbose:
        logging.getLogger('').setLevel(logging.INFO)
    else:
        logging.getLogger('').setLevel(logging.ERROR)

    if args.hostname:
        if 'func' in args:
            args.func(args)
        else:
            raise Exception('Attempt to run in client mode but no command is given')
    else:
        server(args.port,args.perf_iteration)

def real_main():
    parse_args()

def main():
    real_main()
    return
    try:
        real_main()
    except Exception as e:
        logging.error("%s %s" % (e, sys.stderr))
        sys.exit(1)

if __name__ == '__main__':
	main()
