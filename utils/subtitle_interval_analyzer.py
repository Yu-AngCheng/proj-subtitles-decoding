import re
from datetime import datetime


def parse_time(time_str):
    return datetime.strptime(time_str, '%H:%M:%S,%f')


def calculate_durations(srt_file_path):
    with open(srt_file_path, 'r') as file:
        lines = file.readlines()

    durations = []
    current_index = None
    for line in lines:
        # Check for index line
        if line.strip().isdigit():
            current_index = int(line.strip())
        elif '-->' in line:
            time_matches = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', line)
            if len(time_matches) == 2:
                start_time, end_time = time_matches
                duration = (parse_time(end_time) - parse_time(start_time)).total_seconds()
                durations.append((current_index, duration))

    return durations


def find_max_min_avg_intervals(srt_file_path):
    durations = calculate_durations(srt_file_path)
    max_duration = max(durations, key=lambda x: x[1])
    min_duration = min(durations, key=lambda x: x[1])
    avg_duration = sum(duration[1] for duration in durations) / len(durations)

    return max_duration, min_duration, avg_duration


if __name__ == '__main__':
    srt_path = '../data/Skyfall.srt'
    max_duration, min_duration, avg_duration = find_max_min_avg_intervals(srt_path)
    print(f"Max duration: {max_duration[1]}s, idx:{max_duration[0]}")
    print(f"Min duration: {min_duration[1]}s, idx: {min_duration[0]}")
    print(f"Average duration: {avg_duration:.3f}s")
