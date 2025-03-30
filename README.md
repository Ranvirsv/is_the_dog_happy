# is_the_dog_happy

Repository for the Dog emotion detection project

## Setup Instruction

Before you start, make sure you have the kaggle.json file in the current directory <br>
Get the kaggle.json file (Insturctions: https://arc.net/l/quote/reqqlxbn)<br>
<b>NOTE</b>: The downloaded file might be named different, rename to kaggle.json

1. Clone the repository
2. Run `conda env create -f environment.yml` (Pytorch has removed their conda packages, if you run into errors with pytorch, reinstall using pip)
3. Activate the environment `conda activate DOG_EMOTION`
   - <b>NOTE</b>: If you already have the environment but there are changes in the environment.yml file, run `conda env update -f environment.yml --prune`
4. Run the script `get_data.py`
