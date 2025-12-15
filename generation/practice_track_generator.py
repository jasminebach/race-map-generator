import numpy as np
import time

def generate_segment(stype):
    if stype == "slalom":
        x = np.linspace(0, 60, 100)
        y = 6 * np.sin(x / 6)
        return x, y, 0

    if stype == "left_turn":
        R = np.random.uniform(10, 20)
        ang = np.deg2rad(90)
    elif stype == "right_turn":
        R = np.random.uniform(10, 20)
        ang = -np.deg2rad(90)
    else:
        L = np.random.uniform(40, 80)
        return np.linspace(0, L, 40), np.zeros(40), 0

    t = np.linspace(0, ang, 60)
    return R * np.sin(t), R * (1 - np.cos(t)), ang


def generate_practice_track(focus, lap_target=600):
    np.random.seed(int(time.time()))
    x = y = heading = total = 0
    X, Y = [], []

    pool_map = {
        "slalom": ["slalom", "straight"],
        "left_turn": ["left_turn", "straight"],
        "right_turn": ["right_turn", "straight"],
    }

    pool = pool_map.get(focus, ["straight"])

    while total < lap_target:
        stype = np.random.choice(pool)
        xs, ys, dtheta = generate_segment(stype)

        c, s = np.cos(heading), np.sin(heading)
        xr = c * xs - s * ys + x
        yr = s * xs + c * ys + y

        X.append(xr)
        Y.append(yr)

        total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
        x, y = xr[-1], yr[-1]
        heading += dtheta

    return np.hstack(X), np.hstack(Y)


# import numpy as np
# import time

# def generate_segment(stype):
#     if stype == "slalom":
#         x = np.linspace(0, 60, 100)
#         y = 6 * np.sin(x / 6)
#         return x, y, 0

#     if stype == "left_turn":
#         R = np.random.uniform(10, 20)
#         ang = np.deg2rad(90)
#     elif stype == "right_turn":
#         R = np.random.uniform(10, 20)
#         ang = -np.deg2rad(90)
#     else:
#         L = np.random.uniform(40, 80)
#         return np.linspace(0, L, 40), np.zeros(40), 0

#     t = np.linspace(0, ang, 60)
#     return R * np.sin(t), R * (1 - np.cos(t)), ang


# def generate_practice_track(focus, lap_target=600):
#     np.random.seed(int(time.time()))
#     x = y = heading = total = 0
#     X, Y = [], []

#     pool_map = {
#         "slalom": ["slalom", "straight"],
#         "left_turn": ["left_turn", "straight"],
#         "right_turn": ["right_turn", "straight"],
#     }

#     pool = pool_map.get(focus, ["straight"])

#     while total < lap_target:
#         stype = np.random.choice(pool)
#         xs, ys, dtheta = generate_segment(stype)

#         c, s = np.cos(heading), np.sin(heading)
#         xr = c * xs - s * ys + x
#         yr = s * xs + c * ys + y

#         X.append(xr)
#         Y.append(yr)

#         total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
#         x, y = xr[-1], yr[-1]
#         heading += dtheta

#     return np.hstack(X), np.hstack(Y)
