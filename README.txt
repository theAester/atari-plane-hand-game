IMPORTANT: mediapipe only supports python 3.7 to python 3.10 use the provided venv

How to run:
install python 3.10.10. don't add it to PATH
python -m pip install virtualenv
the action script proveded in venv/Scripts/ to activate the venv:
  on linux do: source ./venv/Scripts/activate
  on windows open command prompt (not powershell) and do: C:\Users\<user>\path\to\test-hand\venv\Scripts\activate.bat

after activating the virtual env run python -m pip install -r requirements.txt

then use collect_closed_data.py and collect_open_data.py to collect closed hand and open hand data respectively
after this you should have the file 'closed.pickle' and 'open.pickle'. at this point run collect_data.py to collect these two into a labeled dataset. you should now have the file 'data.pickle'

at this stage you may run train_model.py to train the Support Vector classifier, after training is complete you will see the accuracy score and the file model.pickle is created

the model is ready, run action.py to see an example of the usage``
