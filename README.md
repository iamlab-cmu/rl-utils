# rl-utils
A library for reinforcement learning algorithms, including REPS.

# Installation

## Optional: Create and source virtual environment

### Create directory of virtual environments
`mkdir -p ~/envs`

### Create virtual environment for this application using Python3. You may replace `rl_utils` with your choice of virtual environment name.
`virtualenv -p /usr/bin/python3 ~/envs/rl_utils`

### Activate virtual environment
`source ~/envs/rl_utils/bin/activate`

## Install package with requirements
`pip install . -r requirements.txt`

# Test
All unit tests should pass. Run unit tests through the following command:
`pytest`
