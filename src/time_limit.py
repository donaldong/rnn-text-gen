from datetime import datetime, timedelta


def time_limit(
        days=0,
        seconds=0,
        microseconds=0,
        milliseconds=0,
        minutes=0,
        hours=0,
        weeks=0
):
    start_time = datetime.now()
    delta = timedelta(
        days, seconds, milliseconds,
        minutes, hours, weeks
    )
    while delta > datetime.now() - start_time:
        yield
