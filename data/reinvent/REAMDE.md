Clone reinvent 
`git clone git@bitbucket.org:gerolab/reinvent.git`

Change to dev branch
`git checkout docking_pkfk`



Use reinvent venv `. /data/venv/reinvent/bin/activate`

Change path to priors `prior_file` and `agent_file` in toml config `toml/ESR1.toml`. 

Create directory for a protein and go to it `mkdir ESR1 && cd ESR1`

Train reinvent for a target `reinvent ../toml/ESR1_train.toml`

Sample molecules `reinvent ../toml/ESR1_sample.toml`. This will create `sampling_ESR1.csv`
