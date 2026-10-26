import tkinter as tk
import math
import random
import json
import os
import heapq


ROOM_W = 22.0
ROOM_H = 15.0
ROBOT_R = 0.35
GOAL_R = 0.9
GOAL_VIS_R = 0.55
PERSON_R = 0.32

FOV = math.radians(90)
N_VISION_RAYS = 12
MAX_DEPTH = 26.0

DT = 0.12
MOVE_STEP = 2.6 * DT
TURN_STEP = 2.5 * DT
N_SUB = 4

ACTIONS = ["FORWARD", "LEFT", "RIGHT"]
QTABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable.json")
QTABLE_FORMAT = "room3d-v1"

DIFF_NAMES = ["Empty room", "Furniture", "Moving people"]


VAL_SEEDS = [9000 + i for i in range(6)]


def wrap_angle(a):
    
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def point_rect_dist(px, py, rect):
    
    x, y, w, h = rect
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return math.hypot(dx, dy)


def ray_rect(ox, oy, dx, dy, x0, y0, x1, y1):
    
    tmin = 0.0
    tmax = float("inf")
    axis = -1
    if abs(dx) < 1e-9:
        if ox < x0 or ox > x1:
            return None
    else:
        ta = (x0 - ox) / dx
        tb = (x1 - ox) / dx
        if ta > tb:
            ta, tb = tb, ta
        if ta > tmin:
            tmin = ta
            axis = 0
        if tb < tmax:
            tmax = tb
        if tmin > tmax:
            return None
    if abs(dy) < 1e-9:
        if oy < y0 or oy > y1:
            return None
    else:
        ta = (y0 - oy) / dy
        tb = (y1 - oy) / dy
        if ta > tb:
            ta, tb = tb, ta
        if ta > tmin:
            tmin = ta
            axis = 1
        if tb < tmax:
            tmax = tb
        if tmin > tmax:
            return None
    if tmax < 0.0:
        return None
    hx = ox + dx * tmin
    hy = oy + dy * tmin
    if axis == 0:
        u = (hy - y0) / max(1e-9, (y1 - y0))
    else:
        u = (hx - x0) / max(1e-9, (x1 - x0))
    return (tmin, axis, u)


def ray_circle(ox, oy, dx, dy, cx, cy, r):
    
    lx = cx - ox
    ly = cy - oy
    tca = lx * dx + ly * dy
    d2 = lx * lx + ly * ly - tca * tca
    r2 = r * r
    if d2 > r2:
        return None
    thc = math.sqrt(r2 - d2)
    t = tca - thc
    if t < 0.0:
        t = tca + thc
    if t < 0.0:
        return None
    return t


def segment_blocked(x0, y0, x1, y1, rects, inflate):
    
    dx = x1 - x0
    dy = y1 - y0
    for (rx, ry, rw, rh) in rects:
        hit = ray_rect(x0, y0, dx, dy,
                       rx - inflate, ry - inflate,
                       rx + rw + inflate, ry + rh + inflate)
        if hit is not None and hit[0] <= 1.0:
            return True
    return False


def circle_rect_push(px, py, radius, rect):
    
    x, y, w, h = rect
    cx = clamp(px, x, x + w)
    cy = clamp(py, y, y + h)
    ddx = px - cx
    ddy = py - cy
    d2 = ddx * ddx + ddy * ddy
    if d2 > radius * radius:
        return None
    if d2 > 1e-12:
        d = math.sqrt(d2)
        push = (radius - d) / d
        return (ddx * push, ddy * push)

    left = px - x
    right = (x + w) - px
    down = py - y
    up = (y + h) - py
    m = min(left, right, down, up)
    if m == left:
        return (-(left + radius), 0.0)
    if m == right:
        return (right + radius, 0.0)
    if m == down:
        return (0.0, -(down + radius))
    return (0.0, up + radius)


class World:
    

    def __init__(self, difficulty=1, seed=None):
        self.difficulty = difficulty
        rng = random.Random(seed) if seed is not None else random.Random()
        self.furniture = []
        self.people = []
        self.goal_x = ROOM_W - 2.0
        self.goal_y = ROOM_H - 2.0
        self.start_x = 2.0
        self.start_y = 2.0
        self.start_a = 0.0
        self._generate(rng)


    def _layout_ok(self):
        
        path, reached = plan_path(self, self.start_x, self.start_y,
                                  self.goal_x, self.goal_y,
                                  avoid_people=False)
        return reached and len(path) >= 2

    def _generate(self, rng):

        for _ in range(200):
            gx = rng.uniform(2.0, ROOM_W - 2.0)
            gy = rng.uniform(2.0, ROOM_H - 2.0)
            sx = rng.uniform(1.5, ROOM_W - 1.5)
            sy = rng.uniform(1.5, ROOM_H - 1.5)
            if math.hypot(gx - sx, gy - sy) >= 11.0:
                self.goal_x, self.goal_y = gx, gy
                self.start_x, self.start_y = sx, sy
                break
        n_blocks = {0: 0, 1: 6, 2: 4}[self.difficulty]
        placed = []
        tries = 0
        while len(placed) < n_blocks and tries < 300:
            tries += 1
            w = rng.uniform(1.6, 4.2)
            h = rng.uniform(1.2, 3.0)
            if rng.random() < 0.4:
                w, h = h, w
            x = rng.uniform(1.6, ROOM_W - 1.6 - w)
            y = rng.uniform(1.6, ROOM_H - 1.6 - h)
            rect = (x, y, w, h)

            if point_rect_dist(self.goal_x, self.goal_y, rect) < 2.3:
                continue
            if point_rect_dist(self.start_x, self.start_y, rect) < 1.8:
                continue

            clash = False
            for other in placed:
                if (x - 0.75 < other[0] + other[2] and other[0] - 0.75 < x + w and
                        y - 0.75 < other[1] + other[3] and other[1] - 0.75 < y + h):
                    clash = True
                    break
            if clash:
                continue
            placed.append(rect)
        self.furniture = placed
        if not self._layout_ok():
            self._fallback_layout()

        if self.difficulty >= 2:
            self.people = []
            for _ in range(3):
                for _try in range(100):
                    px = rng.uniform(1.2, ROOM_W - 1.2)
                    py = rng.uniform(1.2, ROOM_H - 1.2)
                    if math.hypot(px - self.goal_x, py - self.goal_y) < 3.0:
                        continue
                    if math.hypot(px - self.start_x, py - self.start_y) < 2.2:
                        continue
                    if any(point_rect_dist(px, py, r) < 0.5 for r in self.furniture):
                        continue
                    ang = rng.uniform(0.0, 2.0 * math.pi)
                    spd = rng.uniform(1.0, 1.5)
                    self.people.append({"x": px, "y": py,
                                        "vx": math.cos(ang) * spd,
                                        "vy": math.sin(ang) * spd,
                                        "r": PERSON_R})
                    break

        want = math.atan2(self.goal_y - self.start_y, self.goal_x - self.start_x)
        self.start_a = want + rng.uniform(-0.4, 0.4)

    def _fallback_layout(self):
        
        self.furniture = []
        if self.difficulty >= 1:
            self.furniture = [(5.0, 9.5, 3.2, 1.6),
                              (10.0, 3.5, 1.8, 3.2),
                              (15.0, 9.8, 3.4, 1.6),
                              (18.5, 3.0, 1.6, 2.6)]
            if math.hypot(self.goal_x - self.start_x,
                          self.goal_y - self.start_y) < 11.0:
                self.start_x, self.start_y = 2.0, 2.0
                self.goal_x, self.goal_y = ROOM_W - 2.0, ROOM_H - 2.0


    def update_people(self, dt):
        
        for p in self.people:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            r = p["r"]
            if p["x"] < r + 0.15:
                p["x"] = r + 0.15
                p["vx"] = abs(p["vx"])
            if p["x"] > ROOM_W - r - 0.15:
                p["x"] = ROOM_W - r - 0.15
                p["vx"] = -abs(p["vx"])
            if p["y"] < r + 0.15:
                p["y"] = r + 0.15
                p["vy"] = abs(p["vy"])
            if p["y"] > ROOM_H - r - 0.15:
                p["y"] = ROOM_H - r - 0.15
                p["vy"] = -abs(p["vy"])
            for rect in self.furniture:
                push = circle_rect_push(p["x"], p["y"], r + 0.05, rect)
                if push is not None:
                    p["x"] += push[0]
                    p["y"] += push[1]

                    pl = math.hypot(push[0], push[1])
                    if pl > 1e-9:
                        nx, ny = push[0] / pl, push[1] / pl
                        dot = p["vx"] * nx + p["vy"] * ny
                        if dot < 0.0:
                            p["vx"] -= 2.0 * dot * nx
                            p["vy"] -= 2.0 * dot * ny

            dx = p["x"] - self.goal_x
            dy = p["y"] - self.goal_y
            d = math.hypot(dx, dy)
            if d < 2.4:
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                nx, ny = dx / d, dy / d
                p["x"] = self.goal_x + nx * 2.4
                p["y"] = self.goal_y + ny * 2.4
                dot = p["vx"] * nx + p["vy"] * ny
                if dot < 0.0:
                    p["vx"] -= 2.0 * dot * nx
                    p["vy"] -= 2.0 * dot * ny


    def cast_ray(self, ox, oy, angle):
        
        dx = math.cos(angle)
        dy = math.sin(angle)
        best_t = MAX_DEPTH
        best = ("none", 0.0, -1)

        ts = []
        if dx > 1e-9:
            ts.append(((ROOM_W - ox) / dx, 0))
        elif dx < -1e-9:
            ts.append(((0.0 - ox) / dx, 0))
        if dy > 1e-9:
            ts.append(((ROOM_H - oy) / dy, 1))
        elif dy < -1e-9:
            ts.append(((0.0 - oy) / dy, 1))
        ts = [t for t in ts if t[0] > 0.0]
        if ts:
            t, axis = min(ts)
            if t < best_t:
                hx = ox + dx * t
                hy = oy + dy * t
                u = ((hy if axis == 0 else hx) / 3.0) % 1.0
                best_t = t
                best = ("wall", u, axis)
        for (x, y, w, h) in self.furniture:
            hit = ray_rect(ox, oy, dx, dy, x, y, x + w, y + h)
            if hit is not None and hit[0] < best_t:
                best_t = hit[0]
                best = ("furniture", hit[2], hit[1])
        for p in self.people:
            t = ray_circle(ox, oy, dx, dy, p["x"], p["y"], p["r"])
            if t is not None and t < best_t:
                best_t = t
                best = ("person", 0.5, -1)
        t = ray_circle(ox, oy, dx, dy, self.goal_x, self.goal_y, GOAL_VIS_R)
        if t is not None and t < best_t:
            best_t = t
            best = ("goal", 0.5, -1)
        return (best_t, best[0], best[1], best[2])

    def goal_los_clear(self, ox, oy):
        
        if segment_blocked(ox, oy, self.goal_x, self.goal_y,
                           self.furniture, 0.05):
            return False
        for p in self.people:

            vx = self.goal_x - ox
            vy = self.goal_y - oy
            wx = p["x"] - ox
            wy = p["y"] - oy
            vv = vx * vx + vy * vy
            tt = 0.0 if vv < 1e-12 else clamp((wx * vx + wy * vy) / vv, 0.0, 1.0)
            cx = ox + vx * tt
            cy = oy + vy * tt
            if math.hypot(p["x"] - cx, p["y"] - cy) < p["r"] + 0.05:
                return False
        return True


def sense(world, x, y, a):
    
    depths = []
    kinds = []
    for i in range(N_VISION_RAYS):

        ang = a + FOV / 2.0 - FOV * i / (N_VISION_RAYS - 1)
        d, kind, _u, _ax = world.cast_ray(x, y, ang)
        depths.append(d)
        kinds.append(kind)
    left_d = sum(depths[0:4]) / 4.0
    mid_d = sum(depths[4:8]) / 4.0
    right_d = sum(depths[8:12]) / 4.0

    def obs_mean(ds, ks):


        vals = [d for d, k in zip(ds, ks) if k != "goal"]
        if not vals:
            return MAX_DEPTH
        return sum(vals) / len(vals)

    left_obs = obs_mean(depths[0:4], kinds[0:4])
    mid_obs = obs_mean(depths[4:8], kinds[4:8])
    right_obs = obs_mean(depths[8:12], kinds[8:12])
    gx = world.goal_x - x
    gy = world.goal_y - y
    dist = math.hypot(gx, gy)
    bearing = wrap_angle(math.atan2(gy, gx) - a)
    visible = 0
    if abs(bearing) < FOV / 2.0 and dist <= 13.0 and world.goal_los_clear(x, y):
        visible = 1

    def bucket(v):
        return 0 if v < 1.2 else (1 if v < 3.5 else 2)

    if abs(bearing) > 2.6:
        sector = 3
    elif bearing > 0.3:
        sector = 0
    elif bearing < -0.3:
        sector = 2
    else:
        sector = 1
    dist_bucket = 0 if dist < 5.0 else (1 if dist < 12.0 else 2)
    state = (visible, sector, bucket(left_d), bucket(mid_d),
             bucket(right_d), dist_bucket)
    return {"depths": depths, "kinds": kinds,
            "left_d": left_d, "mid_d": mid_d, "right_d": right_d,
            "left_obs": left_obs, "mid_obs": mid_obs, "right_obs": right_obs,
            "visible": visible, "bearing": bearing, "dist": dist,
            "state": state}


GRID_CELL = 0.5
GRID_W = int(math.ceil(ROOM_W / GRID_CELL))
GRID_H = int(math.ceil(ROOM_H / GRID_CELL))


def _cell_blocked(world, cx, cy, inflate, avoid_people):
    
    px = (cx + 0.5) * GRID_CELL
    py = (cy + 0.5) * GRID_CELL
    if px < ROBOT_R + 0.05 or px > ROOM_W - ROBOT_R - 0.05:
        return True
    if py < ROBOT_R + 0.05 or py > ROOM_H - ROBOT_R - 0.05:
        return True
    for rect in world.furniture:
        if point_rect_dist(px, py, rect) < inflate:
            return True
    if avoid_people:
        for p in world.people:
            if math.hypot(px - p["x"], py - p["y"]) < p["r"] + ROBOT_R + 0.15:
                return True
    return False


def plan_path(world, sx, sy, gx, gy, avoid_people=True):
    
    inflate = ROBOT_R + 0.2
    blocked = [[_cell_blocked(world, cx, cy, inflate, avoid_people)
                for cx in range(GRID_W)] for cy in range(GRID_H)]

    def nearest_free(px, py):
        ccx = clamp(int(px / GRID_CELL), 0, GRID_W - 1)
        ccy = clamp(int(py / GRID_CELL), 0, GRID_H - 1)
        if not blocked[ccy][ccx]:
            return (ccx, ccy)
        for rad in range(1, 6):
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    nx, ny = ccx + dx, ccy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and not blocked[ny][nx]:
                        return (nx, ny)
        return (ccx, ccy)

    start = nearest_free(sx, sy)
    goal = nearest_free(gx, gy)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came = {start: None}
    gscore = {start: 0.0}
    visited_best = start
    visited_best_d = math.hypot(start[0] - goal[0], start[1] - goal[1])
    reached = False
    while open_heap:
        _f, cur = heapq.heappop(open_heap)
        if cur == goal:
            reached = True
            break
        d_goal = math.hypot(cur[0] - goal[0], cur[1] - goal[1])
        if d_goal < visited_best_d:
            visited_best_d = d_goal
            visited_best = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if blocked[ny][nx]:
                continue
            ng = gscore[cur] + 1.0
            if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = cur
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(open_heap, (ng + h, (nx, ny)))
    target = goal if reached else visited_best
    cells = []
    cur = target
    while cur is not None:
        cells.append(cur)
        cur = came.get(cur)
    cells.reverse()
    pts = [(sx, sy)] + [((cx + 0.5) * GRID_CELL, (cy + 0.5) * GRID_CELL)
                         for (cx, cy) in cells[1:]]
    if reached:
        pts.append((gx, gy))

    smooth = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and segment_blocked(smooth[-1][0], smooth[-1][1],
                                            pts[j][0], pts[j][1],
                                            world.furniture, ROBOT_R + 0.05):
            j -= 1
        smooth.append(pts[j])
        i = j
    if len(smooth) > 1 and math.hypot(smooth[1][0] - sx, smooth[1][1] - sy) < 0.3:
        smooth.pop(1)
    return smooth, reached


class Supervisor:
    

    PROGRESS_WINDOW = 120
    ESCAPE_GRACE = 60

    def __init__(self):
        self.best_dist = None
        self.since_progress = 0
        self.mode = "policy"
        self.escape_seq = []
        self.escape_used = False
        self.path = None
        self.wp_idx = 0
        self.replan_in = 0
        self.handoffs = 0
        self.rescues = 0
        self.visits = {}

    def reset_episode(self):
        self.__init__()


    def decide(self, ep, sn):
        if self.mode == "escape":
            if self.escape_seq:
                return self.escape_seq.pop(0)
            self.mode = "policy"
            self.since_progress = self.PROGRESS_WINDOW - self.ESCAPE_GRACE
        if self.mode == "rescue":
            act = self._rescue_action(ep, sn)
            if act is not None:
                return act

            self.mode = "policy"
            self.since_progress = self.PROGRESS_WINDOW - 20

        cell = (int(ep.x / 0.75), int(ep.y / 0.75))
        self.visits[cell] = self.visits.get(cell, 0) + 1
        looped = self.visits[cell] >= 8 and sn["dist"] > GOAL_R + 0.5
        if self.since_progress >= self.PROGRESS_WINDOW or looped:
            if not self.escape_used:
                self._start_escape(sn)
                if self.escape_seq:
                    return self.escape_seq.pop(0)
            self._start_rescue(ep)
            act = self._rescue_action(ep, sn)
            if act is not None:
                return act
        return None

    def _start_escape(self, sn):
        
        self.escape_used = True
        self.mode = "escape"
        turn = "LEFT" if sn["left_obs"] >= sn["right_obs"] else "RIGHT"
        self.escape_seq = [turn] * 10
        if sn["mid_obs"] > 1.0:
            self.escape_seq += ["FORWARD"] * 5

    def _start_rescue(self, ep):
        self.mode = "rescue"
        self.rescues += 1
        self.path = None
        self.wp_idx = 0
        self.replan_in = 0

    def _replan(self, ep):
        path, reached = plan_path(ep.world, ep.x, ep.y,
                                  ep.world.goal_x, ep.world.goal_y,
                                  avoid_people=True)
        if len(path) < 2:
            path, reached = plan_path(ep.world, ep.x, ep.y,
                                      ep.world.goal_x, ep.world.goal_y,
                                      avoid_people=False)
        self.path = path
        self.wp_idx = 1 if len(path) > 1 else 0
        self.replan_in = 30

    def _rescue_action(self, ep, sn):
        if self.path is None or self.wp_idx >= len(self.path) or self.replan_in <= 0:
            self._replan(ep)
        self.replan_in -= 1
        if self.path is None or len(self.path) < 2:
            return None

        if (self.handoffs < 3 and sn["visible"] and sn["dist"] < 3.5
                and sn["mid_obs"] > 1.6):
            self.handoffs += 1
            self.path = None
            return None
        wx, wy = self.path[self.wp_idx]
        while (self.wp_idx < len(self.path) - 1
               and math.hypot(wx - ep.x, wy - ep.y) < 0.55):
            self.wp_idx += 1
            wx, wy = self.path[self.wp_idx]
        if math.hypot(wx - ep.x, wy - ep.y) < 0.55 and self.wp_idx >= len(self.path) - 1:


            if self.handoffs >= 3:
                wx, wy = ep.world.goal_x, ep.world.goal_y
            else:
                self.handoffs += 1
                self.path = None
                return None
        beta = wrap_angle(math.atan2(wy - ep.y, wx - ep.x) - ep.a)


        person_close = any(k == "person" and d < 1.0
                           for k, d in zip(sn["kinds"][4:8], sn["depths"][4:8]))
        if beta > 0.28:
            return "LEFT"
        if beta < -0.28:
            return "RIGHT"
        if person_close and sn["mid_obs"] < 0.9:
            return "LEFT" if beta >= 0 else "RIGHT"
        if sn["mid_obs"] < 0.55:
            return "LEFT" if sn["left_obs"] >= sn["right_obs"] else "RIGHT"
        return "FORWARD"


    def after_step(self, dist):
        if self.best_dist is None or dist < self.best_dist - 0.2:
            self.best_dist = dist
            self.since_progress = 0
            if self.mode == "policy":
                self.escape_used = False
        else:
            self.since_progress += 1


def rules_action(ep, sn):
    
    mem = ep.policy_mem
    if mem.get("avoid", 0) > 0:
        mem["avoid"] -= 1
        return mem["avoid_dir"]
    beta = sn["bearing"]


    if sn["visible"]:
        if beta > 0.26:
            return "LEFT"
        if beta < -0.26:
            return "RIGHT"
        return "FORWARD"

    if sn["mid_obs"] < 1.0:
        mem["avoid"] = 4
        mem["avoid_dir"] = "LEFT" if sn["left_obs"] > sn["right_obs"] else "RIGHT"
        return mem["avoid_dir"]

    if abs(beta) < 0.25 and sn["mid_obs"] > 1.5:
        return "FORWARD"
    return "LEFT" if beta > 0 else "RIGHT"


class Episode:
    

    def __init__(self, world, policy="rules", agent=None, supervisor=True,
                 max_steps=1000):
        self.world = world
        self.policy = policy
        self.agent = agent
        self.use_supervisor = supervisor
        self.sup = Supervisor() if supervisor else None
        self.max_steps = max_steps
        self.x = world.start_x
        self.y = world.start_y
        self.a = world.start_a
        self.prev_x, self.prev_y, self.prev_a = self.x, self.y, self.a
        self.people_prev = [(p["x"], p["y"]) for p in world.people]
        self.steps = 0
        self.collisions = 0
        self.reward = 0.0
        self.path_len = 0.0
        self.done = False
        self.reached = False
        self.action = "READY"
        self.status = "Ready"
        self.policy_mem = {}
        self.sense = sense(world, self.x, self.y, self.a)
        self.prev_dist = self.sense["dist"]
        self.optimal = self.prev_dist
        if self.sup is not None:
            self.sup.after_step(self.prev_dist)

    def _apply(self, action):
        
        world = self.world
        self.people_prev = [(p["x"], p["y"]) for p in world.people]
        world.update_people(DT)
        collided = False
        sub_dt = DT / N_SUB
        if action == "LEFT":
            self.a = wrap_angle(self.a + TURN_STEP)
        elif action == "RIGHT":
            self.a = wrap_angle(self.a - TURN_STEP)
        elif action == "FORWARD":
            for _ in range(N_SUB):
                nx = self.x + math.cos(self.a) * (MOVE_STEP / N_SUB)
                ny = self.y + math.sin(self.a) * (MOVE_STEP / N_SUB)

                cx = clamp(nx, ROBOT_R, ROOM_W - ROBOT_R)
                cy = clamp(ny, ROBOT_R, ROOM_H - ROBOT_R)
                if cx != nx or cy != ny:
                    collided = True
                nx, ny = cx, cy

                for rect in world.furniture:
                    push = circle_rect_push(nx, ny, ROBOT_R, rect)
                    if push is not None:
                        nx += push[0]
                        ny += push[1]
                        collided = True

                for p in world.people:
                    dx = nx - p["x"]
                    dy = ny - p["y"]
                    d = math.hypot(dx, dy)
                    need = ROBOT_R + p["r"]
                    if d < need:
                        if d < 1e-6:
                            dx, dy, d = 1.0, 0.0, 1.0
                        nx = p["x"] + dx / d * need
                        ny = p["y"] + dy / d * need
                        collided = True
                nx = clamp(nx, ROBOT_R, ROOM_W - ROBOT_R)
                ny = clamp(ny, ROBOT_R, ROOM_H - ROBOT_R)
                self.path_len += math.hypot(nx - self.x, ny - self.y)
                self.x, self.y = nx, ny
        return collided

    def step(self, explore_eps=None):
        
        if self.done:
            return None
        self.prev_x, self.prev_y, self.prev_a = self.x, self.y, self.a
        sn = self.sense
        state = sn["state"]
        override = None
        if self.sup is not None:
            override = self.sup.decide(self, sn)
        if override is not None:
            action = override
        elif self.policy == "learned" and self.agent is not None and self.agent.table:
            if explore_eps is not None:
                ai = self.agent.act(state, explore_eps)
            else:
                ai = self.agent.greedy(state)
            action = ACTIONS[ai]
        else:
            action = rules_action(self, sn)
        self.action = action
        collided = self._apply(action)
        self.steps += 1
        if collided:
            self.collisions += 1
        self.sense = sense(self.world, self.x, self.y, self.a)
        dist = self.sense["dist"]

        r = 2.0 * (self.prev_dist - dist) - 0.1
        if collided:
            r -= 10.0
        if dist <= GOAL_R:
            self.reached = True
            self.done = True
            r += 100.0
            self.action = "GOAL"
        self.reward += r
        self.prev_dist = dist
        if self.sup is not None:
            self.sup.after_step(dist)
        if self.steps >= self.max_steps:
            self.done = True

        if self.reached:
            self.status = "Reached the goal"
        elif self.sup is not None and self.sup.mode == "rescue":
            self.status = "Rescue path active"
        elif self.sup is not None and self.sup.mode == "escape":
            self.status = "Getting unstuck"
        elif action == "FORWARD":
            self.status = "Going to the goal"
        elif sn["mid_obs"] < 1.2:
            self.status = "Avoiding obstacle"
        else:
            self.status = "Turning toward the goal"
        return {"state": state, "action": ACTIONS.index(action),
                "reward": r, "next_state": self.sense["state"],
                "done": self.done}

    def spl(self):
        if not self.reached:
            return 0.0
        return min(1.0, self.optimal / max(self.path_len, self.optimal))


class QAgent:
    

    def __init__(self, alpha=0.2, gamma=0.98):
        self.table = {}
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _key(state):
        return ",".join(map(str, state))

    def q(self, state):
        return self.table.setdefault(self._key(state), [0.0] * len(ACTIONS))

    def greedy(self, state):
        qs = self.q(state)
        best = 0
        for i in range(1, len(qs)):
            if qs[i] > qs[best]:
                best = i
        return best

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(len(ACTIONS))
        return self.greedy(state)

    def update(self, s, a, r, s2, done):
        qs = self.q(s)
        target = r if done else r + self.gamma * max(self.q(s2))
        qs[a] += self.alpha * (target - qs[a])

    def save(self, path=QTABLE_PATH):
        try:
            with open(path, "w") as f:
                json.dump({"format": QTABLE_FORMAT, "q": self.table}, f)
            return True
        except OSError:
            return False

    def load(self, path=QTABLE_PATH):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("format") == QTABLE_FORMAT:
                raw = data.get("q", {})

                self.table = {k: [float(x) for x in v] for k, v in raw.items()
                              if isinstance(v, list) and len(v) == len(ACTIONS)}
                return True
        except (OSError, ValueError):
            pass
        return False


def run_training_episode(agent, difficulty, eps, max_steps=300):
    
    w = World(difficulty=difficulty)
    ep = Episode(w, policy="learned", agent=agent, supervisor=False,
                 max_steps=max_steps)
    while not ep.done:
        tr = ep.step(explore_eps=eps)
        if tr is None:
            break
        agent.update(tr["state"], tr["action"], tr["reward"],
                     tr["next_state"], tr["done"])
    return ep.reached, ep.reward, ep


def raw_eval(policy, difficulty, seeds, agent=None, max_steps=300):
    
    n = len(seeds)
    ok = 0
    col = 0
    stp = 0
    for seed in seeds:
        w = World(difficulty=difficulty, seed=seed)
        ep = Episode(w, policy=policy, agent=agent, supervisor=False,
                     max_steps=max_steps)
        while not ep.done:
            ep.step()
        if ep.reached:
            ok += 1
        col += ep.collisions
        stp += ep.steps
    return {"episodes": n, "success_rate": ok / n,
            "avg_collisions": col / n, "avg_steps": stp / n}


def validation_score(agent):
    
    res = {d: raw_eval("learned", d, VAL_SEEDS, agent=agent)
           for d in (0, 1, 2)}
    mean_sr = sum(res[d]["success_rate"] for d in (0, 1, 2)) / 3
    mean_col = sum(res[d]["avg_collisions"] for d in (0, 1, 2)) / 3
    mean_steps = sum(res[d]["avg_steps"] for d in (0, 1, 2)) / 3
    return (res[0]["success_rate"], mean_sr, -mean_col, -mean_steps), res


class QTrainer:
    

    VALIDATE_EVERY = 100

    def __init__(self, agent, total_episodes):
        self.agent = agent
        self.total = total_episodes
        self.ep = 0

        self.rng = random.Random(random.randrange(1 << 30))
        self.recent_r = []
        self.recent_ok = []
        self.hist = []
        self.last_stage = 0
        self.last_eps = 1.0
        self.last_episode = None
        self.best_score = None
        self.best_table = None
        self.best_ep = 0
        self.last_val = None

    def _difficulty(self):
        
        f = self.ep / max(1, self.total)
        if f < 0.30:
            probs = (0.60, 0.30, 0.10)
        elif f < 0.65:
            probs = (0.30, 0.45, 0.25)
        else:
            probs = (0.25, 0.40, 0.35)
        r = self.rng.random()
        if r < probs[0]:
            return 0
        if r < probs[0] + probs[1]:
            return 1
        return 2

    def _epsilon(self):
        return max(0.05, 1.0 - 0.95 * min(1.0, self.ep / (0.75 * self.total)))

    def _validate(self):
        score, res = validation_score(self.agent)
        self.last_val = (score, res)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_table = {k: list(v) for k, v in self.agent.table.items()}
            self.best_ep = self.ep

    def run_episodes(self, n):
        
        for _ in range(n):
            if self.ep >= self.total:
                break
            stage = self._difficulty()
            eps = self._epsilon()

            self.agent.alpha = 0.2 - 0.1 * (self.ep / max(1, self.total))
            ok, r, ep_obj = run_training_episode(self.agent, stage, eps)
            self.last_stage = stage
            self.last_eps = eps
            self.last_episode = ep_obj
            self.recent_r.append(r)
            self.recent_ok.append(1 if ok else 0)
            if len(self.recent_r) > 50:
                self.recent_r.pop(0)
                self.recent_ok.pop(0)
            self.ep += 1
            if self.ep % 10 == 0 or self.ep == self.total:
                self.hist.append((self.ep, self.avg_reward(), self.success_rate()))
            if self.ep % self.VALIDATE_EVERY == 0 or self.ep == self.total:
                self._validate()
        if self.ep >= self.total:
            self.finish()
        return self.ep >= self.total

    def finish(self):
        
        if self.best_table is not None:
            self.agent.table = {k: list(v) for k, v in self.best_table.items()}

    def avg_reward(self):
        return sum(self.recent_r) / max(1, len(self.recent_r))

    def success_rate(self):
        return sum(self.recent_ok) / max(1, len(self.recent_ok))


def train_fresh(total_episodes, seed=None):
    
    if seed is not None:
        random.seed(seed)
    agent = QAgent()
    trainer = QTrainer(agent, total_episodes)
    while trainer.ep < total_episodes:
        trainer.run_episodes(25)
    return agent, trainer


def run_eval(policy, difficulty, seeds, agent=None, supervisor=True,
             max_steps=1000):
    
    n = len(seeds)
    ok = 0
    col = 0
    stp = 0
    spl = 0.0
    rescues = 0
    rescue_eps = 0
    far = 0
    for seed in seeds:
        w = World(difficulty=difficulty, seed=seed)
        ep = Episode(w, policy=policy, agent=agent, supervisor=supervisor,
                     max_steps=max_steps)
        while not ep.done:
            ep.step()
        if ep.reached:
            ok += 1
            spl += ep.spl()
        else:
            if ep.sense["dist"] > 2.0:
                far += 1
        col += ep.collisions
        stp += ep.steps
        if ep.sup is not None:
            rescues += ep.sup.rescues
            if ep.sup.rescues > 0:
                rescue_eps += 1
    return {"episodes": n, "reached": ok, "success_rate": ok / n,
            "avg_collisions": col / n, "avg_steps": stp / n,
            "avg_spl": spl / n, "total_rescues": rescues,
            "episodes_with_rescue": rescue_eps, "ended_far": far}


OBSTACLE_TONES = ["#B08954", "#A67C52", "#8D6E63", "#9E7B53", "#7D6A55", "#AD8B5F"]
CAM_W = 648
CAM_H = 432
CAM_COLS = 216
TRAIN_TOTAL = 2400
FRAMES_PER_STEP = 3


def _shade(base, f):
    r, g, b = base
    r = clamp(int(r * f), 0, 255)
    g = clamp(int(g * f), 0, 255)
    b = clamp(int(b * f), 0, 255)
    return "#%02x%02x%02x" % (r, g, b)


def _depth_color(dc):
    
    if dc < 3.0:
        t = dc / 3.0
        c0, c1 = (215, 45, 35), (240, 200, 45)
    else:
        t = clamp((dc - 3.0) / 4.0, 0.0, 1.0)
        c0, c1 = (240, 200, 45), (45, 180, 85)
    return "#%02x%02x%02x" % (int(c0[0] + (c1[0] - c0[0]) * t),
                              int(c0[1] + (c1[1] - c0[1]) * t),
                              int(c0[2] + (c1[2] - c0[2]) * t))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Autonomous Navigation Robot using RL - Prathish S J")
        self.qagent = QAgent()
        self.qagent.load()
        self.policy = "rules"
        self.mode = "demo"
        self.running = False
        self.frame_in_step = 0
        self.done_ticks = 0
        self.render_depths = None

        self.trainer = None
        self.train_agent = None
        self.train_hist = []
        self.train_before = None

        self.eval_seeds = []
        self.eval_idx = 0
        self.eval_acc = None
        self.eval_ep = None

        tk.Label(root, text="Autonomous Navigation Robot using Reinforcement Learning",
                 font=("Calibri", 15, "bold")).pack(pady=(8, 2))

        main = tk.Frame(root)
        main.pack(padx=8)
        left = tk.Frame(main)
        left.grid(row=0, column=0, padx=4, sticky="n")
        right = tk.Frame(main)
        right.grid(row=0, column=1, padx=4, sticky="n")

        self.cam_c = tk.Canvas(left, width=CAM_W, height=CAM_H, bg="black",
                               highlightthickness=1, highlightbackground="black")
        self.cam_c.pack()
        tk.Label(left, text="ROBOT CAMERA VIEW - 3D ROOM (what the robot sees)",
                 font=("Calibri", 9, "bold")).pack()
        self.depth_c = tk.Canvas(left, width=CAM_W, height=46, bg="black",
                                 highlightthickness=1, highlightbackground="black")
        self.depth_c.pack(pady=(4, 0))
        tk.Label(left, text="DEPTH FROM CAMERA RAYS (red = close, green = far)",
                 font=("Calibri", 9, "bold")).pack()

        self.map_c = tk.Canvas(right, width=300, height=205, bg="#EFE9DC",
                               highlightthickness=1, highlightbackground="black")
        self.map_c.pack()
        tk.Label(right, text="TOP VIEW MAP (the whole room)",
                 font=("Calibri", 9, "bold")).pack()
        self.vis_c = tk.Canvas(right, width=300, height=160, bg="white",
                               highlightthickness=1, highlightbackground="black")
        self.vis_c.pack(pady=(4, 0))
        tk.Label(right, text="WHAT THE ROBOT SEES (edges and objects)",
                 font=("Calibri", 9, "bold")).pack()
        self.chart_c = tk.Canvas(right, width=300, height=140, bg="white",
                                 highlightthickness=1, highlightbackground="black")
        self.chart_c.pack(pady=(4, 0))
        tk.Label(right, text="LEARNING CURVE (while training)",
                 font=("Calibri", 9, "bold")).pack()
        self.report = tk.Label(right, text="", font=("Calibri", 9), justify="left",
                               anchor="nw", wraplength=296)
        self.report.pack(fill="x", pady=(4, 0))

        self.stats = tk.Label(root, text="", font=("Calibri", 10), justify="left")
        self.stats.pack(pady=4)

        ctl = tk.Frame(root)
        ctl.pack(pady=2)
        self.btn_start = tk.Button(ctl, text="START", width=9, command=self.start)
        self.btn_start.grid(row=0, column=0, padx=3)
        self.btn_pause = tk.Button(ctl, text="PAUSE", width=9, command=self.pause)
        self.btn_pause.grid(row=0, column=1, padx=3)
        tk.Button(ctl, text="RESET", width=9, command=self.reset).grid(row=0, column=2, padx=3)
        tk.Label(ctl, text="Room type:").grid(row=0, column=3, padx=(14, 2))
        self.diff = tk.StringVar(value="1")
        diff_labels = ["Empty room - no obstacles",
                       "Furniture - blocks in the room",
                       "Moving people - people walking"]
        for i, t in enumerate(diff_labels):
            tk.Radiobutton(ctl, text=t, variable=self.diff, value=str(i),
                           command=self.change_settings).grid(row=0, column=4 + i)
        tk.Label(ctl, text="Who drives:").grid(row=0, column=7, padx=(14, 2))
        self.pol = tk.StringVar(value="rules")
        tk.Radiobutton(ctl, text="Rules - fixed steps written by us", variable=self.pol, value="rules",
                       command=self.change_settings).grid(row=0, column=8)
        tk.Radiobutton(ctl, text="Learned - robot trained itself", variable=self.pol, value="learned",
                       command=self.change_settings).grid(row=0, column=9)

        ctl2 = tk.Frame(root)
        ctl2.pack(pady=2)
        self.btn_train = tk.Button(ctl2, text="TRAIN (learns)", width=16,
                                   command=self.start_training)
        self.btn_train.grid(row=0, column=0, padx=4)
        self.btn_eval = tk.Button(ctl2, text="EVALUATE (test on new rooms)", width=26,
                                  command=self.start_eval)
        self.btn_eval.grid(row=0, column=1, padx=4)
        self.mode_status = tk.Label(ctl2, font=("Calibri", 9), justify="left",
                                    text=self._brain_text())
        self.mode_status.grid(row=0, column=2, padx=8)

        self.explain = tk.Label(root, font=("Calibri", 10, "bold"),
                                text=self._explain_text())
        self.explain.pack(pady=2)

        tk.Label(root, text="Keys: Space = Start/Pause    R = Reset",
                 font=("Calibri", 8), fg="gray").pack(pady=2)
        root.bind("<space>", lambda e: self.toggle_space())
        root.bind("r", lambda e: self.reset())
        root.bind("R", lambda e: self.reset())

        self._draw_cam_bg()
        self.new_episode()
        self.draw()
        self.draw_chart()
        self.loop()

    def _brain_text(self):
        if self.qagent.table:
            return "Learned brain: loaded from qtable.json (%d states)" % len(self.qagent.table)
        return "Learned brain: not trained yet - press TRAIN (learns)"


    def new_episode(self):
        self.policy = self.pol.get()
        w = World(difficulty=int(self.diff.get()))
        self.ep = Episode(w, policy=self.policy, agent=self.qagent,
                          supervisor=True, max_steps=1000)
        self.frame_in_step = 0
        self.done_ticks = 0

    def start(self):
        if self.mode != "demo":
            return
        if self.ep.done:
            self.new_episode()
        self.running = True

    def pause(self):
        self.running = False

    def toggle_space(self):
        if self.mode != "demo":
            return
        if self.running:
            self.pause()
        else:
            self.start()

    def reset(self):
        self.mode = "demo"
        self.running = False
        self.eval_ep = None
        self.new_episode()
        self.draw()

    def _explain_text(self):
        room = {0: "an empty room with nothing inside",
                1: "a room with furniture blocks in the way",
                2: "a room with people walking around"}[int(self.diff.get())]
        if self.pol.get() == "rules":
            who = "a robot driven by fixed rules written by us"
        else:
            who = "a robot that learned to drive by training itself"
        return ("What you are seeing: %s, finding its way "
                "to the green goal in %s." % (who, room))

    def change_settings(self):
        if hasattr(self, "explain"):
            self.explain.config(text=self._explain_text())
        if self.mode != "demo":
            return
        self.running = False
        self.new_episode()
        self.draw()


    def start_training(self):
        if self.mode == "train":
            return
        self.mode = "train"
        self.running = False
        self.train_agent = QAgent()
        self.trainer = QTrainer(self.train_agent, TRAIN_TOTAL)
        self.train_hist = []
        self.mode_status.config(text="Measuring the untrained brain on test rooms...")
        self.root.update_idletasks()
        self.train_before = {d: raw_eval("learned", d, VAL_SEEDS, agent=self.train_agent)
                             for d in (0, 1, 2)}
        self.mode_status.config(text="Training...")

    def _train_chunk(self):
        tr = self.trainer
        done = tr.run_episodes(12)
        self.train_hist = tr.hist
        if tr.last_episode is not None:
            self.ep = tr.last_episode
        text = ("Training the robot brain...  rooms tried: %d of %d\n"
                "Reached the goal in %d of the last 50 tries    average reward %+.1f"
                % (tr.ep, tr.total, int(round(tr.success_rate() * 50)),
                   tr.avg_reward()))
        if tr.last_val is not None:
            score, _res = tr.last_val
            text += ("\nBest brain so far (from try %d): test rooms - empty room "
                     "%d of 6, all rooms %d of 18"
                     % (tr.best_ep, int(round(score[0] * 6)),
                        int(round(score[1] * 18))))
        self.mode_status.config(text=text)
        self.draw_chart()
        if done:
            self._finish_training()

    def _finish_training(self):
        self.mode = "demo"
        after = {d: raw_eval("learned", d, VAL_SEEDS, agent=self.train_agent)
                 for d in (0, 1, 2)}
        self.qagent = self.train_agent
        saved = self.qagent.save()
        lines = ["Training done. Learned brain %s (%d states learned)."
                 % ("saved to qtable.json" if saved else "could NOT be saved",
                    len(self.qagent.table)),
                 "The best brain found during training was kept.",
                 "Test rooms (never trained on), reached the goal:"]
        for d in (0, 1, 2):
            b = int(round(self.train_before[d]["success_rate"] * 6))
            a = int(round(after[d]["success_rate"] * 6))
            lines.append("  %s:  %d of 6  ->  %d of 6" % (DIFF_NAMES[d], b, a))
        lines.append("Driver set to Learned (trained). Press START to watch it drive.")
        self.report.config(text="\n".join(lines))
        self.mode_status.config(text=self._brain_text())
        self.pol.set("learned")
        self.new_episode()


    def start_eval(self):
        if self.mode != "demo":
            return
        self.mode = "eval"
        self.running = False
        self.policy = self.pol.get()
        self.eval_seeds = [random.randrange(1 << 30) for _ in range(20)]
        self.eval_idx = 0
        self.eval_acc = {"ok": 0, "col": 0, "stp": 0, "spl": 0.0, "rescue_eps": 0}
        self._next_eval_episode()
        self.mode_status.config(text="Testing on new rooms...")
        self.report.config(text="")

    def _next_eval_episode(self):
        d = int(self.diff.get())
        w = World(difficulty=d, seed=self.eval_seeds[self.eval_idx])
        self.eval_ep = Episode(w, policy=self.policy, agent=self.qagent,
                               supervisor=True, max_steps=1000)
        self.ep = self.eval_ep

    def _eval_chunk(self):
        ep = self.eval_ep
        for _ in range(8):
            if ep.done:
                break
            ep.step()
        if ep.done:
            acc = self.eval_acc
            if ep.reached:
                acc["ok"] += 1
                acc["spl"] += ep.spl()
            acc["col"] += ep.collisions
            acc["stp"] += ep.steps
            if ep.sup is not None and ep.sup.rescues > 0:
                acc["rescue_eps"] += 1
            self.eval_idx += 1
            n = self.eval_idx
            self.report.config(
                text="TESTING on new rooms  (%s, driver: %s)\n"
                     "room %d of 20 done\n"
                     "reached the goal: %d of %d    bumps: %d    average steps: %.0f"
                     % (DIFF_NAMES[int(self.diff.get())],
                        "Rules" if self.policy == "rules" else "Learned (trained)",
                        n, acc["ok"], n, acc["col"], acc["stp"] / n))
            if self.eval_idx >= len(self.eval_seeds):
                self._finish_eval()
            else:
                self._next_eval_episode()

    def _finish_eval(self):
        acc = self.eval_acc
        total = len(self.eval_seeds)
        self.mode = "demo"
        self.report.config(
            text="TEST DONE  (%s, driver: %s, 20 new rooms)\n"
                 "reached the goal: %d of %d\n"
                 "average bumps per room: %.1f\n"
                 "average steps per room: %.0f\n"
                 "average SPL: %.2f\n"
                 "rescue path was used in %d of the rooms"
                 % (DIFF_NAMES[int(self.diff.get())],
                    "Rules" if self.policy == "rules" else "Learned (trained)",
                    acc["ok"], total, acc["col"] / total, acc["stp"] / total,
                    acc["spl"] / total, acc["rescue_eps"]))
        self.mode_status.config(text="Testing complete.")
        self.new_episode()


    def draw_chart(self):
        c = self.chart_c
        c.delete("all")
        W, H = 300, 140
        pad = 26
        base = H - 16
        c.create_line(pad, 6, pad, base, fill="gray")
        c.create_line(pad, base, W - 6, base, fill="gray")
        c.create_text(pad + 2, 8, text="reached (green)", font=("Calibri", 7),
                      anchor="w", fill="green")
        c.create_text(pad + 78, 8, text="reward (blue)", font=("Calibri", 7),
                      anchor="w", fill="blue")
        c.create_text(W - 8, base + 11, text="tries", font=("Calibri", 7),
                      anchor="e", fill="gray")
        if not self.train_hist:
            c.create_text(W // 2, H // 2 + 8,
                          text="The learning curve appears here\nwhen you press TRAIN (learns)",
                          font=("Calibri", 8), fill="gray", justify="center")
            return
        rewards = [h[1] for h in self.train_hist]
        rmin, rmax = min(rewards), max(rewards)
        if rmax - rmin < 1e-9:
            rmax = rmin + 1.0
        pts_s = []
        pts_r = []
        for (epn, ar, sr) in self.train_hist:
            x = pad + (epn / TRAIN_TOTAL) * (W - pad - 8)
            pts_s += [x, base - sr * (base - 12)]
            pts_r += [x, base - ((ar - rmin) / (rmax - rmin)) * (base - 12)]
        if len(self.train_hist) >= 2:
            c.create_line(pts_s, fill="green", width=2)
            c.create_line(pts_r, fill="blue", width=1)


    def _interp(self, alpha):
        ep = self.ep
        x = ep.prev_x + (ep.x - ep.prev_x) * alpha
        y = ep.prev_y + (ep.y - ep.prev_y) * alpha
        a = ep.prev_a + wrap_angle(ep.a - ep.prev_a) * alpha
        people = []
        for prev, p in zip(ep.people_prev, ep.world.people):
            people.append((prev[0] + (p["x"] - prev[0]) * alpha,
                           prev[1] + (p["y"] - prev[1]) * alpha, p["r"]))
        if len(people) < len(ep.world.people):
            for p in ep.world.people[len(people):]:
                people.append((p["x"], p["y"], p["r"]))
        return x, y, a, people


    def _draw_cam_bg(self):
        c = self.cam_c
        horizon = CAM_H // 2
        bands = 12
        for b in range(bands):
            y0 = horizon * b // bands
            y1 = horizon * (b + 1) // bands
            f = b / bands
            col = _shade((104, 122, 140), 0.35 + 0.65 * f)
            c.create_rectangle(0, y0, CAM_W, y1, fill=col, outline="", tags="bg")
        fbands = 14
        for b in range(fbands):
            y0 = horizon + (CAM_H - horizon) * b // fbands
            y1 = horizon + (CAM_H - horizon) * (b + 1) // fbands
            f = b / fbands
            col = _shade((150, 130, 100), 1.0 - 0.55 * f)
            c.create_rectangle(0, y0, CAM_W, y1, fill=col, outline="", tags="bg")
        c.create_line(0, horizon, CAM_W, horizon, fill="#D8E2E8", width=2, tags="bg")

    def draw_camera(self, alpha):
        c = self.cam_c
        c.delete("dyn")
        ep = self.ep
        world = ep.world
        x, y, a, people = self._interp(alpha)
        horizon = CAM_H // 2
        strip = CAM_W / CAM_COLS
        depths = []
        prev_dc = None
        for i in range(CAM_COLS):
            rel = FOV / 2.0 - FOV * i / (CAM_COLS - 1)
            d, kind, u, axis = world.cast_ray(x, y, a + rel)
            dc = max(d * math.cos(rel), 0.2)
            depths.append(d)
            h = min(CAM_H * 2.3 / dc, CAM_H * 1.9)
            if kind == "wall":
                base = (122, 134, 150)
                tex = 0.95 if (u * 6.0) % 2.0 < 1.0 else 1.0
            elif kind == "furniture":
                base = (158, 113, 66)
                tex = 0.90 if (u * 8.0) % 2.0 < 1.0 else 1.0
            elif kind == "person":
                base = (216, 120, 40)
                tex = 1.0
            elif kind == "goal":
                base = (30, 205, 95)
                tex = 1.0
            else:
                base = (90, 90, 90)
                tex = 1.0
            bright = 1.0 / (1.0 + 0.16 * dc)
            side = 0.78 if axis == 0 else 1.0
            col = _shade(base, bright * side * tex)
            if prev_dc is not None and abs(dc - prev_dc) > 0.45:
                col = _shade(base, bright * side * tex * 0.7)
            prev_dc = dc
            ytop = horizon - h / 2.0
            c.create_rectangle(i * strip, ytop, i * strip + strip + 1,
                               horizon + h / 2.0, fill=col, outline="",
                               tags="dyn")
        self.render_depths = depths

        gx = world.goal_x - x
        gy = world.goal_y - y
        gdist = math.hypot(gx, gy)
        beta = wrap_angle(math.atan2(gy, gx) - a)
        if abs(beta) < FOV / 2.0 and gdist > 0.3:
            half = math.atan2(GOAL_VIS_R, gdist)
            ci0 = int((FOV / 2.0 - (beta + half)) / FOV * (CAM_COLS - 1))
            ci1 = int((FOV / 2.0 - (beta - half)) / FOV * (CAM_COLS - 1))
            ci0 = clamp(ci0, 0, CAM_COLS - 1)
            ci1 = clamp(ci1, 0, CAM_COLS - 1)
            hp = min(CAM_H * 2.3 / max(gdist * math.cos(beta), 0.3),
                     CAM_H * 1.9) * 0.82
            drawn = False
            for ci in range(ci0, ci1 + 1):
                if gdist < depths[ci] + 0.35:
                    c.create_rectangle(ci * strip, horizon - hp / 2.0 - 3,
                                       ci * strip + strip + 1,
                                       horizon + hp / 2.0 + 3,
                                       fill="#8CF79B", outline="", tags="dyn")
                    c.create_rectangle(ci * strip + 0.7, horizon - hp / 2.0,
                                       ci * strip + strip + 0.3,
                                       horizon + hp / 2.0,
                                       fill="#17C964", outline="", tags="dyn")
                    drawn = True
            if drawn:
                cxpix = (ci0 + ci1) / 2.0 * strip
                c.create_text(cxpix, horizon - hp / 2.0 - 14, text="GOAL",
                              fill="#17C964", font=("Calibri", 10, "bold"),
                              tags="dyn")

        for (px, py, pr) in people:
            dx = px - x
            dy = py - y
            pd = math.hypot(dx, dy)
            pb = wrap_angle(math.atan2(dy, dx) - a)
            if abs(pb) >= FOV / 2.0 or pd < 0.3:
                continue
            ci = int((FOV / 2.0 - pb) / FOV * (CAM_COLS - 1))
            ci = clamp(ci, 0, CAM_COLS - 1)
            if pd >= depths[ci] + 0.3:
                continue
            dpc = max(pd * math.cos(pb), 0.3)
            wall_h = min(CAM_H * 2.3 / dpc, CAM_H * 1.9)
            ph = wall_h * 0.72
            pw = ph * 0.40
            cxpix = (FOV / 2.0 - pb) / FOV * CAM_W
            bottom = horizon + wall_h / 2.0
            top = bottom - ph
            c.create_rectangle(cxpix - pw / 2, top + ph * 0.24,
                               cxpix + pw / 2, bottom, fill="#E67E22",
                               outline="#A04000", tags="dyn")
            hr = pw * 0.30
            c.create_oval(cxpix - hr, top, cxpix + hr, top + 2 * hr,
                          fill="#F5CBA7", outline="#A04000", tags="dyn")

        c.create_line(CAM_W / 2, horizon - 10, CAM_W / 2, horizon + 10,
                      fill="yellow", tags="dyn")
        c.create_line(CAM_W / 2 - 10, horizon, CAM_W / 2 + 10, horizon,
                      fill="yellow", tags="dyn")

    def draw_depth(self):
        c = self.depth_c
        c.delete("all")
        if not self.render_depths:
            return
        strip = CAM_W / CAM_COLS
        for i, d in enumerate(self.render_depths):
            rel = FOV / 2.0 - FOV * i / (CAM_COLS - 1)
            dc = d * math.cos(rel)
            c.create_rectangle(i * strip, 0, i * strip + strip + 1, 46,
                               fill=_depth_color(dc), outline="")
        c.create_line(CAM_W / 2, 0, CAM_W / 2, 46, fill="black")


    def draw_map(self, alpha):
        c = self.map_c
        c.delete("all")
        ep = self.ep
        world = ep.world
        Wpx, Hpx = 300, 205
        s = Wpx / ROOM_W

        def sx(wx):
            return wx * s

        def sy(wy):
            return Hpx - wy * s

        c.create_rectangle(2, 2, Wpx - 2, Hpx - 2, outline="#4A3F33", width=5)
        for idx, (fx, fy, fw, fh) in enumerate(world.furniture):
            tone = OBSTACLE_TONES[idx % len(OBSTACLE_TONES)]
            c.create_rectangle(sx(fx) + 2, sy(fy + fh) + 3, sx(fx + fw) + 2,
                               sy(fy) + 3, fill="#C9BBA3", outline="")
            c.create_rectangle(sx(fx), sy(fy + fh), sx(fx + fw), sy(fy),
                               fill=tone, outline="#5D4A38")

        if ep.sup is not None and ep.sup.mode == "rescue" and ep.sup.path:
            pts = []
            for (wx, wy) in ep.sup.path:
                pts += [sx(wx), sy(wy)]
            if len(pts) >= 4:
                c.create_line(pts, fill="#2471A3", width=2, dash=(4, 2))

        gx, gy = sx(world.goal_x), sy(world.goal_y)
        c.create_oval(gx - 10, gy - 10, gx + 10, gy + 10, fill="white",
                      outline="#27AE60", width=3)
        c.create_oval(gx - 5, gy - 5, gx + 5, gy + 5, fill="#27AE60",
                      outline="#1E8449")
        x, y, a, people = self._interp(alpha)
        for (px, py, pr) in people:
            c.create_oval(sx(px) - 6, sy(py) - 4, sx(px) + 8, sy(py) + 8,
                          fill="#C9BBA3", outline="")
            c.create_oval(sx(px) - 6, sy(py) - 6, sx(px) + 6, sy(py) + 6,
                          fill="#F39C12", outline="#B9770E")
        rx, ry = sx(x), sy(y)

        for sgn in (-1, 1):
            wa = -(a + sgn * FOV / 2.0)
            c.create_line(rx, ry, rx + math.cos(wa) * 30, ry + math.sin(wa) * 30,
                          fill="#AED6F1")
        fx, fy = math.cos(a), -math.sin(a)
        pxv, pyv = math.sin(a), math.cos(a)
        for sgn in (-1, 1):
            wx = rx + pxv * 8 * sgn
            wy = ry + pyv * 8 * sgn
            pts = [(wx + fx * 4 - pxv * 2, wy + fy * 4 - pyv * 2),
                   (wx + fx * 4 + pxv * 2, wy + fy * 4 + pyv * 2),
                   (wx - fx * 4 + pxv * 2, wy - fy * 4 + pyv * 2),
                   (wx - fx * 4 - pxv * 2, wy - fy * 4 - pyv * 2)]
            c.create_polygon(pts, fill="#2E2E2E", outline="")
        c.create_oval(rx - 8, ry - 8, rx + 8, ry + 8, fill="#F2F2F2",
                      outline="#333333", width=2)
        nose = [(rx + fx * 12, ry + fy * 12),
                (rx + pxv * 4, ry + pyv * 4),
                (rx - pxv * 4, ry - pyv * 4)]
        c.create_polygon(nose, fill="#566573", outline="")
        c.create_oval(rx + fx * 5 - 2.5, ry + fy * 5 - 2.5,
                      rx + fx * 5 + 2.5, ry + fy * 5 + 2.5,
                      fill="#3498DB", outline="")


    def draw_vision(self):
        c = self.vis_c
        c.delete("all")
        W, H = 300, 160
        ep = self.ep
        world = ep.world
        n = 48
        depths = []
        for i in range(n):
            ang = ep.a + FOV / 2.0 - FOV * i / (n - 1)
            d, _k, _u, _ax = world.cast_ray(ep.x, ep.y, ang)
            depths.append(d)
        strip = W / n

        pts = [0, H - 14]
        for i, d in enumerate(depths):
            hh = (1.0 - min(d, 12.0) / 12.0) * (H - 40)
            pts += [i * strip, H - 14 - hh]
        pts += [W, H - 14]
        c.create_polygon(pts, fill="#D6EAF8", outline="#2471A3")
        prev = depths[0]
        for i, d in enumerate(depths):
            if abs(d - prev) > 0.8:
                c.create_line(i * strip, 8, i * strip, H - 14, fill="black", width=2)
            prev = d
        for xx in (W / 3, 2 * W / 3):
            c.create_line(xx, 0, xx, H, fill="lightgray", dash=(3, 3))
        sn = ep.sense
        if sn["visible"]:
            frac = 0.5 - sn["bearing"] / FOV
            bx = clamp(frac * W - 26, 2, W - 54)
            c.create_rectangle(bx, 26, bx + 52, 74, outline="green", width=2)
            c.create_text(bx + 26, 16, text="GOAL SEEN", fill="green",
                          font=("Calibri", 8, "bold"))
        if sn["mid_obs"] < 1.2:
            c.create_rectangle(W / 2 - 26, 60, W / 2 + 26, 108, outline="red", width=2)
            c.create_text(W / 2, 50, text="OBSTACLE CLOSE", fill="red",
                          font=("Calibri", 8, "bold"))
        c.create_text(W / 2, H - 5, text="black lines = edges found in the camera rays",
                      font=("Calibri", 7), fill="gray")


    def draw(self, alpha=1.0):
        self.draw_camera(alpha)
        self.draw_depth()
        self.draw_map(alpha)
        self.draw_vision()
        ep = self.ep
        status = ep.status
        if ep.done and not ep.reached:
            status = "Stopped - step limit reached"
        self.stats.config(
            text="Steps: %d    Distance to goal: %.1f    Collisions: %d    "
                 "Robot is doing: %s"
                 % (ep.steps, ep.sense["dist"], ep.collisions, status))

    def loop(self):
        if self.mode == "train":
            self._train_chunk()
            self.draw(alpha=1.0)
        elif self.mode == "eval":
            self._eval_chunk()
            self.draw(alpha=1.0)
        elif self.running and not self.ep.done:
            self.frame_in_step += 1
            if self.frame_in_step >= FRAMES_PER_STEP:
                self.frame_in_step = 0
                self.ep.step()
            self.draw(alpha=self.frame_in_step / FRAMES_PER_STEP)
        elif self.ep.done and self.running and self.mode == "demo":

            self.done_ticks += 1
            self.draw(alpha=1.0)
            if self.done_ticks > 28:
                self.new_episode()
        else:
            self.draw(alpha=1.0)
        self.root.after(33, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
