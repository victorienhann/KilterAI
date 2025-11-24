from src.utils.Roles import ROLES as roles

def parse_frames(frames):
    frames = str(frames.values)
    i = 0
    start = []
    middle = []
    finish = []
    foot = []
    while(frames[i] != 'p'):
        i = i + 1
    while i < len(frames) - 2:
        p = int(frames[i+1:i+5])
        i = i + 5
        r = int(frames[i+1:i+3])
        i = i + 3
        if r in roles["start"]:
            start.append(p)
        elif r in roles["middle"]:
            middle.append(p)
        elif r in roles["finish"]:
            finish.append(p)
        elif r in roles["foot"]:
            foot.append(p)
        else :
            raise ValueError("Invalid role {}".format(r))


    return start, middle, finish, foot