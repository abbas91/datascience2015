# -- Login to Azure Linux box with SSH key -- #

# - check if you have the folder containing ssh keys
cd ~/.ssh/
# - Generate new keys 
ssh-keygen -t rsa -b 2048 -C "myusername@myserver"
"mark@instance1" # - label
"Will ask you give a file name? & add password to it?"
# - check is "id_rsa" and "id_rsa.pub" SSH key pair in "~/.ssh" dir
ls -al ~/.ssh
# cat rsa key for paste in public key
cat ~/.ssh/id_rsa.pub
# add the created key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
# - create and configure SSH config file
touch ~/.ssh/config
vi ~/.ssh/config
# ------ example file ---------- #
# Azure Keys
Host test
  Hostname 23.96.118.253
  User mark
# ./Azure Keys
# Default Settings
Host *
  PubkeyAuthentication=yes
  IdentitiesOnly=yes
  ServerAliveInterval=60
  ServerAliveCountMax=30
  ControlMaster auto
  ControlPath ~/.ssh/SSHConnections/ssh-%r@%h:%p # make sure the folder exists - ~/.ssh/SSHConnections
  ControlPersist 4h
  IdentityFile ~/.ssh/id_rsa
# -------- end of file ----------- #

# Access server using ssh
ssh test_instance





/anaconda3/bin/python -c "import IPython;print(IPython.lib.passwd())"

/usr/local/etc/jupyter/jupyter_notebook_config.py #parameter name "c.NotebookApp.password"
"sha1:xxxxxxx"
/etc/init.d/jupyter



