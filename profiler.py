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
import re

class Profiling:
    def __init__(self):
        self.terminate = threading.Condition()
        self.is_active = False
        self.events = self.get_perf_power_events()
        self.timeseries = {}

    def get_perf_power_events(self):
        events = []
        result = subprocess.run(['perf', 'list'], stdout=subprocess.PIPE)
        for l in result.stdout.decode('utf-8').splitlines():
            l = l.lstrip()
            m = re.match("(power/energy-.*/)\s*\[Kernel PMU event]", l)
            if m:
                events.append(m.group(1))
        return events

    def sample_perf_stat_power(self):
        events_str = ','.join(self.events)
        cmd = ['sudo', 'perf', 'stat', '-a', '-e', events_str, 'sleep', '1']
        result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out = result.stdout.decode('utf-8').splitlines() + result.stderr.decode('utf-8').splitlines()
        for e in self.events:
            for l in out:
                l = l.lstrip()
                m = re.match("(.*)\s+.*\s+{}".format(e), l)
                if m:
                    value = m.group(1)
                    self.timeseries.setdefault(e, []).append(float(value))

profiling = Profiling()

def profile_thread():
    while profiling.is_active:
        profiling.sample_perf_stat_power()

        profiling.terminate.acquire()
        profiling.terminate.wait(timeout=1)
        profiling.terminate.release()

def start():
    global profiling
    profiling.state_usage_start = power_state_usage()
    profiling.state_time_start = power_state_time()        
    profiling.is_active=True
    x = threading.Thread(target=profile_thread)    
    x.daemon = True
    x.start()

def stop():
    global profiling
    profiling.is_active=False
    profiling.terminate.acquire()
    profiling.terminate.notify()
    profiling.terminate.release()

    profiling.state_usage_stop = power_state_usage()
    profiling.state_time_stop = power_state_time()        

def power_state_names():
    state_names = []
    stream = os.popen('ls /sys/devices/system/cpu/cpu0/cpuidle/')
    states = stream.readlines()
    for state in states:
        state = state.strip()
        stream = os.popen("cat /sys/devices/system/cpu/cpu0/cpuidle/{}/name".format(state))
        state_names.append(stream.read().strip())
    return state_names

def power_state_metric(metric):
    state_names = power_state_names()
    cpu_state_values = []
    for cpu in range(0, os.cpu_count()):
        state_values = []
        for state in range(0, len(state_names)):
            output = open("/sys/devices/system/cpu/cpu{}/cpuidle/state{}/{}".format(cpu, state, metric)).read()
            state_values.append(int(output))
        cpu_state_values.append(state_values)
    return cpu_state_values

def power_state_usage():
    return power_state_metric('usage')

def power_state_time():
    return power_state_metric('time')

def power_state_diff(new_vector, old_vector):
    diff = []
    for (new, old) in zip(new_vector, old_vector):
        diff.append([x[0] - x[1] for x in zip(new, old)])
    return diff
    
def report():
    global profiling
    timeseries = profiling.timeseries
    usage = []
    usage.append(power_state_names())
    usage.append(power_state_diff(profiling.state_usage_stop, profiling.state_usage_start))
    time = []
    time.append(power_state_names())
    time.append(power_state_diff(profiling.state_time_stop, profiling.state_time_start))
    return [timeseries, usage, time]

def server(port):
    hostname = socket.gethostname().split('.')[0]
    server = SimpleXMLRPCServer((hostname, port), allow_none=True)
    print("Listening on port {}...".format(port))
    server.register_function(start, "start")
    server.register_function(stop, "stop")
    server.register_function(report, "report")
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

    @staticmethod
    def action(args):
        with xmlrpc.client.ServerProxy("http://{}:{}/".format(args.hostname, args.port)) as proxy:
            timeseries = proxy.report()
            print(timeseries)

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

    subparsers = parser.add_subparsers(dest='subparser_name', help='sub-command help')
    actions = [StartAction, StopAction, ReportAction]
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
        server(args.port)

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
