import matplotlib.pyplot as plt
import re
import pdb
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


def plot_durations(srt_file_path):
    durations = calculate_durations(srt_file_path)
    duration_times = [item[1] for item in durations]

    plt.figure(figsize=(16, 6))
    plt.hist(duration_times, bins=551)
    plt.title('Duration Distribution')
    plt.xlabel('Duration Time')
    plt.ylabel('Frequency')

    custom_ticks = [0.5 * i for i in range(int(max(duration_times)//0.5) + 2)]
    plt.xticks(custom_ticks)

    plt.savefig('../figures/duration_distribution.png', dpi=500)
    plt.show()  

    return

if __name__ == '__main__':
    srt_path = '../data/Skyfall.srt'
    plot_durations(srt_path)
    # max_duration, min_duration, avg_duration = find_max_min_avg_intervals(srt_path)
    # print(f"Max duration: {max_duration[1]}s, idx:{max_duration[0]}")
    # print(f"Min duration: {min_duration[1]}s, idx: {min_duration[0]}")
    # print(f"Average duration: {avg_duration:.3f}s")
