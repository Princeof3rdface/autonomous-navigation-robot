# Autonomous Navigation Robot using Reinforcement Learning

A robot in a room, a goal it has to reach, and no map. The robot sees the room
only through its own camera rays and drives itself to the goal without hitting
the furniture or the people walking in the room.

This is a simulation project built for the Foundations of Computer Vision
course. There is no hardware - everything runs in one Python file.

## What the robot does

- The room is drawn in 3D from the robot camera using raycasting (216 rays).
- For driving, the robot uses 12 camera rays: how far things are on the left,
  middle and right, whether the goal is in view, and where the goal is.
- It picks one action at a time: FORWARD, LEFT or RIGHT.
- The goal is a green charging dock. Coming within 0.9 units of it counts as
  reached.

## Two ways of driving

- **Rules** - fixed steps written by hand: turn towards the goal when it is
  seen, turn to the freer side when something is close in front.
- **Learned (trained)** - Q-learning. The robot practised 2400 times in fresh
  rooms and learned from rewards: +100 for reaching the goal, -10 for a bump,
  -0.1 per step, and a small reward for moving closer. What it learned is
  saved in `qtable.json` and loaded again at start.

A safety supervisor watches both drivers. If the robot stops making progress
or starts looping, it first makes an escape turn, and if that is not enough a
rescue path is planned with A* search and the robot follows it to the goal.
So the robot does not stay stuck.

## Room types

- **Empty room** - no obstacles
- **Furniture** - blocks in the room
- **Moving people** - people walking around the room

## How to run

You only need Python 3. Nothing to install - the program uses only the
standard library (tkinter, math, random, json, os, heapq).

1. Download `run.py` and `qtable.json` and keep them in the same folder.
   The program looks for its trained brain next to the code file.
2. Double click `run.py`, or run it from a terminal:

```
python run.py
```

3. Press START to watch it drive. TRAIN runs the full learning again and
   saves a new brain. EVALUATE tests the chosen driver on 20 brand new
   rooms. Keys: Space = start/pause, R = reset.

## Files in this repo

- `run.py` - the whole project in one file
- `qtable.json` - the trained brain (439 learned states)
- `Autonomous_Navigation_Robot_Report_Final.pdf` - full written report:
  sensing, both drivers, training, pseudocode, results and limitations
- `Autonomous_Navigation_Robot_Demo_Voice.mp4` - demo video with voice
  narration: both drivers driving, live training, and evaluation

## Results

Tested on 30 brand new rooms for every setting, with the full system
(supervisor on):

| Driver | Empty room | Furniture | Moving people |
|---|---|---|---|
| Rules | 30 of 30 | 30 of 30 | 30 of 30 |
| Learned (trained) | 30 of 30 | 30 of 30 | 30 of 30 |

Honest note: without the supervisor, the learned driver alone is weaker,
mainly in furniture rooms (16 of 30). The report explains this openly.

## Limitations

- Simulation only - no real robot, camera or room.
- People are simple moving shapes that walk in straight lines.
- The learned brain is a Q-table over a small state, not a deep network.
