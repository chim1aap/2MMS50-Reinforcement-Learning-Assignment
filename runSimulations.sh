#!/bin/zsh

/usr/bin/python qlearning.py  &
/usr/bin/python sarsa.py &
/usr/bin/python tdlearning.py &
echo "simulations started!"
