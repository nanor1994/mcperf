#!/bin/bash

#execute  telnet for memcached processing time
{ echo "stats"; sleep 1; } | telnet node1 11211


