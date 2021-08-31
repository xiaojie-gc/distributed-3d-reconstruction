from datetime import datetime


def logger(CURRENT="00000"):
    now = datetime.now()  # current date and time
    return now.strftime("%H:%M:%S") + "[LOG-\033[1m{}\033[0m] ".format(CURRENT)


if __name__ == '__main__':
    s = "[009]-[015]"
    print(s[1:4])
