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











# Create Putty Key SSH access #

# Manue: https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-linux-use-ssh-windows


# - Download "PuTTYGen"
http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html

# - Download "PuTTY"
http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html


# - Open "PuTTYGen"

" 2.For Type of key to generate, select SSH-2 RSA, and then click Generate. "
" 3.Move the mouse around in the area below the progress bar, until the bar fills. Moving the mouse generates random data that is used to generate the key. "
" Once the key has been generated, the public key will be displayed. "
" 4.For added security, you can enter a passphrase in the Key passphrase field, and then type the same value in the Confirm passphrase field."
" 5.Click Save private key to save the key to a .ppk file. This key will be used to authenticate to your Linux-based HDInsight cluster. "
" 6.Click Save public key to save the key as a .txt file. This allows you to reuse the public key in the future when you create additional Linux-based HDInsight clusters."


# - Open "PuTTY"

" 2.If you provided an SSH key when you created your user account, you must perform the following step to select the private key to use when authenticating to the cluster: "

" In Category, expand Connection, expand SSH, and select Auth. Finally, click Browse and select the .ppk file that contains your private key. "

" 3.In Category, select Session. From the Basic options for your PuTTY session screen, enter the SSH address of your HDInsight server in the Host name (or IP address) field. Port: 22"
" There are two possible SSH addresses you may use when connecting to a cluster: "


"               Head node address: To connect to the head node of the cluster, use your cluster name, then -ssh.azurehdinsight.net. For example, mycluster-ssh.azurehdinsight.net. "


"               Edge node address: If you are connecting to an R Server on HDInsight cluster, you can connect to the R Server edge node using the address RServer.CLUSTERNAME.ssh.azurehdinsight.net, "
"               where CLUSTERNAME is the name of your cluster. For example, RServer.mycluster.ssh.azurehdinsight.net. "



" 4.To save the connection information for future use, enter a name for this connection under Saved Sessions, and then click Save. The connection will be added to the list of saved sessions. "


" 5.Click Open to connect to the cluster. "




























# Open IPython Notebook
/anaconda3/bin/python -c "import IPython;print(IPython.lib.passwd())"

/usr/local/etc/jupyter/jupyter_notebook_config.py #parameter name "c.NotebookApp.password"
"sha1:xxxxxxx"
/etc/init.d/jupyter



