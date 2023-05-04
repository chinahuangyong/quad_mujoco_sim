#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../msgs/java
export CLASSPATH=${DIR}/../msgs/java/my_types.jar
pwd
lcm-spy