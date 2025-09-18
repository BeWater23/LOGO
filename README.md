The script is designed to do a Leave One Group Out Analysis on combinatorial catalyst/substrate MLR models.
It interfaces and is compatible with the general MLR function (mlr_utils.py) of the Sigman Lab.
The usual environment of the Sigman Lab python-modeling should be compatible with this repository. In case there are any problems the attached logo_env.yml can be used to create a fresh environemnt.
For cluster usage any submit script can be used. On the University of Utah CHPC (notchpeak) the submit_cli.sh inside this repository can be used:


submit_cli.sh python logo.py