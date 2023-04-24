## How to run this app locally

(The following instructions are for unix-like systems)

Clone this repository and navigate to the directory containing this `README` in
a terminal.

Create and activate a virtual environment (recommended):

```bash
conda create -n sosame python=3.7
conda activate sosame
```

Install the requirements

```bash
pip install -r requirements.txt
```

Run the app. An IP address where you can view the app in your browser will be
displayed in the terminal.

```bash
python app.py
```

For windows try also

```bash
conda install -c anaconda cairo
```