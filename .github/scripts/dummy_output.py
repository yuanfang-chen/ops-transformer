#!/usr/bin/env python3

import datetime
import time


def main() -> None:
    for i in range(1, 41):
        timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        print(f"dummy output {i} at {timestamp}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
