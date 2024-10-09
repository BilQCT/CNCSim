NUM_MEASUREMENTS = 7
MEASUREMENT_SEQ_START_LINE = 23 # BEWARE: 0-indexed
MEASUREMENT_SEQ_PERIOD = 30

def read_specific_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    branches = set()
    for i in range(MEASUREMENT_SEQ_START_LINE, len(lines), MEASUREMENT_SEQ_PERIOD):
        branch_lines = lines[i:i+NUM_MEASUREMENTS]
        branch = []
        for line in branch_lines:
            measurement_line = line.strip().split('|')
            if len(measurement_line) > 1:
                measurment = measurement_line[1].strip()
                branch.append(measurment)
        branches.add(' -> '.join(branch))
    
    return branches

# Example usage
file_path = 'Compilation_data.txt'
branches = read_specific_lines(file_path)

for branch in branches:
    print(branch)