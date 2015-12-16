## <<<< Basic Unix commandline >>>> ##

## -------------------------- Dir Structure ------------------------------ ##
/ # root dir - where everything begins
/bin # contains binaries (Program) must for system to run
/boot # contains linux kernal - initial RAM disk image (boot loaders)
/dev # contains all devices node - all files
/etc # all system wide configuration files / also some shell scripts for each boot time
/home # each user has a home dir (User can only write in their home dir)
/lib # contains shared libraries files used by core system
/lost+found # each formatted partition and devices will have this dir for system failure, usually empyty
/media # for removable media - USB, CD-ROM, appears at insertion
/mnt # for old system, for mount manually for removable devices
/opt # for installing "optional" commerical software
/proc # Not an actual dir, virual file system that contains files for how a linux kernal sees your ocmputer
/root # the home dir for the root account
/sbin # contains programs for vitual system work, for super user only
/tmp # contains temporary files, empty self each time reboot system
/usr # Contains all programs and support files used by regular users, largest dir
/usr/bin # contains excuatible programs installed by your linux distribution
/usr/lib # The shared libraries for the programs in "/usr/bin"
/usr/local # contains programs not included with distribution but are intended for system-wide use are installed.
/usr/sbin # contains more system adminstration programs
/usr/share # contains all shared data used by programs in "/usr/bin", like default configuration files, background, sound files
/usr/share/doc # packages installed's dosumentations files
/var # contains data likely to change, database, user mails, etc
/var/log # contains log file records activities (Must be super user for security reason)



## -------------------------- commands content --------------------------------------- ##































## ---------------------------- Basic Shell ----------------------------- ##
# [1] terminal emulator
[username@machinename]$ # user
[username@machinename]# # super user
# simple commands
$date # current date/time
$cal # calendard
$df # current free space on disk
$free # display free memory
$exit # end session/close

# [2] navigation - tree like structure; home dir is the user dir
# root directory, parent directory, current working directory
$pwd # display current wd
$ls # list all content on that dir
$cd # change current dir
    (1) $cd # home dir
        $cd - # previous dir
        $cd ~ # home dir
            ~user # home dir of that user
            . # working dir
            .. # parent dir
        $cd /user/bin # navigate to that dir (absolute path)
        $cd ./bin #navigate to that dir (relative path)

# [3] exploring the system
$ls /user/bin /user/bin2 # display items for multiple dirs
    (1) $ls -l # more detail/long format
            # long format = [-][rw-r--r--] [1] [root] [root] [2342342] [2013-05-04 11:05] [xxxx.txt] -> [xxxx2.txt]
            # 1.file type/ 2.access right(owner,group,general)/ 3.# of hard link/ 4.username of the file owner/ 5.file size/ 6.last modified time/ 6. file name -> soft link               
        $ls -a # list all even hiden file
        $ls -d # see dir rather than content
        $ls -F # append indicator by end of each list name
        $ls -h # display file size in human readable format
        $ls -r # display in reverse order
        $ls -S # sort by file size
        $ls -t # sort by modified time
   
$file xxx.txt # determine a file type
$less xxx.txt # view the file (pess Q to quit, up, down to navigate by line, uppage, downpage to navigate by page)
              # "G" to the end of the file; "g" to the beginning; "/ xxx" search the next character; 

# [4] Manipulate files and dir
# wildcards - easily specify file name
* # any character
? # and single character
[abs] # any file has a character in that group of characters
[!abs] # ---------------------- not in ---------------------
[[:class:]] # match by class
      (1) [[:alnum:]] # any alphanumeric character
          [[:alpha:]] # any alphabetic character
          [[:digit:]] # any numeral
          [[:lower:]] # any lower case letter
          [[:upper:]] # any uppercase letter
# example:
* # all file
g* # any file start with g
b*.txt # any file satrt with b and followed by any characters and end with txt
Data??? # any file begining with Data and followed by three characters
[abc]* # any file begining with either a,b,or c
BACKUP.[0-9][0-9][0-9] # begining with "BACKUP." followed by any 3 numbers
[[:upper:]]* # any file begining with upper case
[![:digit:]]* # any file not begining with a numerical
*[[:lower:]123] # any file end with a lower case or 1,2,or 3
# Create Dir
$mkdir dir1 dir2 dir3
# copy file / dir
$cp file1 file2 # copy 1 to 2, if 2 exits overwrite, if not create
$cp -i file1 file2 # --------------------------------------------- and give notice
$cp file1 file2 dir1 # copy file1 and file2 into dir1
$cp dir1/* dir2 # copy all files in dir1 into dir2
$cp -r dir1 dir2 # ------------------------------ if dir2 not exits, create one
      (1) $cp -a # required by copying dir
              -r # required by copying dir
              -i # notice
              -u # only those new or new version files
              -v # display informative message
# Move and rename file
$mv file1 file2 # same as cp but orginial file gone
      (1) $mv -i # notice
          $mv -u # as above
          $mv -v # as above
# remove file / dir
$rm file1 file2 
      (1) $rm -i # same
          $rm -r # same
          $rm -v # same
          $rm -f # Ignore any notice
# Create links 
$ln file1 link # -hard links
$ln -s file1 link # -soft links

# [5] unix Command type *** (Check documentaion)
# type1: An executable program
# type2: An command builtin in the shell itself
# type3: An shell function
# type4: An alias (function we defined)
$type command # display a command's type
$which command # display a executable's location (Only for executable, not builtin and aliases)
$help command # get help menu for shell builtins ex. explain ...
$command --help # also get help for that command, usage information
$man 1 command # user command
$man 2 command # Program interface (kernal system)
$man 3 command # Program interface (clibrary) 
$man 4 command # special files
$man 5 command # file format
$man 6 command # games
$man 7 command # miscellaneais
$man 8 command # system adminstration commands
$info command # up, down to see manu pages
# Create your own commands with alias
$alias command_name = 'commands'
$command_name # vanish after session end

# [6] Redirection
# Input/output redirection - connect multiple commands to make powerful command pipeline
# Regular way
Keyborad -> 
        input "stdin" -> 
                 command -> 
                    output "stdout" - result / "stderr" - status/error -> 
                                                              to screen (Not disk file)
# Redirect standrad output
$ls -l /user/bin > xxx.txt # redirect result of "ls" to the file "xxx.txt" *if file not exists, error -> emplty file
$> file1 # another way to create a new file
$ls -l /user/bin >> xxx.txt
$ls -l /user/bin >> xxx.txt # the xxx.txt file should be 2 times the size since ">>" not rewrite as ">"
$ls -l /user/bin 2> xxx.txt # file descriper - 0 = input, 1 = output, 2 = error/status
$ls -l /user/bin > xxx.txt 2>&1 # both output and error (old version)
$ls -l /user/bin &> xxx.txt # both output and error (new version)
$ls -l /user/bin 2> /dev/null # disposing unwanted result - direct it to /dev/null
# Redirect standrad input
$cat file1 file2 > xxx.txt # read mulitple files and redirct them to one file (Usually use to join files)
$cat > xxx.txt # type input to save into the file, "CTRL-D" as typing
# Pipeline - use one command's stdout as another command's stdin
# example:
$ls -l /user/bin | less # display result page by page
$ls -l /user/bin | sort | unique | wc # sort the result -> get unique filename -> count the words
$ls -l /user/bin | sort | unique | grep zip # --------------------------------- -> find the file contains "zip"
$head -n 5 xxx.txt # see first 5 lines
$tail -n 5 xxx.txt # see last 5 lines
      -f /user/bin/xxx.txt # view file in real time


# [7] Expansion - type command , press ENTER - bash performs serveral processes upon the text before it excute command
$echo D*; $echo *S # pathname expansion
$echo ~; $echo ~mark # Tilde expansion
$echo I have $((5*2)) daller # Arithemetic expansion (+ - * / % **)
$echo xxx-{1,2,3}-xxx; $echo a{A{1,2}}, B{3,4}}b # Brace expansion - 1. xxx-1-xxx, xxx-2-xxx, xxx-3-xxx; 2.aA1b, aA2b, aB3b, aB4b
$echo $USER # parameter expansion - system stores small chunk of data, given each a name - "me"
$echo $(ls); $file $(ls /user/bin/* | grep zip) # command substitution - Allow to use the output of a command as an expansion
# Quoting "" '' - escape
"" # escape, except parameter, arithmetic, commend substition 
'' # everything escape
\$; \!; \& # escape special marks


# [8] Manage permission - unix is multi-task, multi-user system
# [-] [rw-] [rw-] [r--] -> from "ls"
# 1.file type: "-" regular; "d" directory; "l" a symbolic link; "c" a character special file; "b" a block special file
# 2.owner; 3.group; 4.general -> r = read; w = write; x = excute (file.r - open)(file.w - write/delete)(file.x = program execute)
#                                                                (dir.r - a list)(dir.w - write/rename)(dir.x - allow to enter) 
$id # display user id
$chmod 600 xxx.txt # change file's mode
#                    0 - ---; 1 - --x; 2 - -w-; 3 - -wx; 4 - r--; 5 - r-x; 6 - rw-; 7 - rwx
$umask 002 # reset default file mode when it is created
# change user identity
# 1.login in another user; 2.$su; 3.$sudo
$su -l mark2 # load a new env - $# , ask for password
$su - # load a new env (super user), ask for password
$su -c "command" # excute a single commend as other user
$sudo command # no load new env (super user), ask for password 
# change file ownership
$sudo chown owner-user xxx.txt
$sudo chown owner-user:owner-group xxx.txt
# change password - for yourself; for others(super user)
$passwd user # ask for old passwd, then insert new passwd
$passwd -S user #username, [L-locked passwd, NP-no passwd, P-usable passwd], date of last changes, expire min age, max age, expire warn period, inactivity period of passwd


# [9] processes - PID - "Process ID"
# manage program waiting their term for CPU (very quick) -> [1] 28364 = job number PID
$ps # view current process
$top # display tasks in real time
$jobs # list active jobs
$bg # place a job in the backend
$fg # place a job in the frontend
$kill PID # send a signal to a process - PID to end
$killall # kill process by names
$shutdown # shutdown reboot system




## ---------------------------- Environment & Configuration ----------------------------- ##


## [1] Environment - A body of information during our shell session (two types of data - environment var/ shell var)
$printenv # Only display envornment variable
$set # Display both shell & environment variables
$echo $variable # View content of a single variable
$alias # Display only aliases
# Environment established
# 1.the "bash" program starts and reads a series of configuration scripts call "startup files" - default env for all users
# 2.More files in user's HOME dir define user's personal environment
# Two kinds of shell sessions: <1> a login shell session (ask passwords) / <2> a non-login shell session 
# Start files for "Login shell session"
.file # hiden file need "ls -a" to view
/etc/profile # a global configuration file applies to all users
~/.bash_profile # a user's personal startup file
~/.bash_login # if "~/.bash_profile" not found, bash templet to read this
~/.profile # if both above not found, bash reads this
# Start files for "non-Login shell session"
/etc/bash.bashrc # a global configuration file applies to all users
~/.bashrc # a user's personal startup file
# - See start file
PATH=$PATH:$HOME/bin # Expansion -> HOME = multiple... ;System know where to find command - "ls"
$export file # make the content avaiable for child processes of this shell
# Modifying environment
$cp .bashrc .bashrc.bak # before modifying start file, create a back-up file
$sudo vi .bashrc # modifying
$source .bashrc # file only load when start session, so we need manually load using "source"


## [2] vi editor
$vi # see versions of vi
# exit
:q! #no save / quit
:wq #saved / quit
:w #just save
# create a file
$vi new_file.txt
# insert mode
i #Start current position
a #Start one position behind
o #the line below the current line
O #the line above the current line
# Navigate the cursor
0 #To beginning of current line
SHIFT-4 #To the end of current line
w #To the next word / punctuation
W #To the next word
b #To the previous word / punctuation
B #To the previous word
"PAGE DOWN" #Down one page
"PAGE UP" #Up one page
SHIFT-G #To the list line of the file
gg #To the beginning of the file
# Delete text (Also cuts)
x #deleting one position
dd #deleting current line
5dd #deleting 5 lines
dw #current position to the end of current word
d$ #current position to the end of current line
d0 #current position to the beginning of current line
dG #current line to end of the file
d20G #current line to the 20th line
# undo
u
# cutting, copying, pasting
yy #copy current line
5yy #copy 5 lines
y$ #current position to the end of the line
y0 #current position to the beginning of the line
yG #current position to the end of the file
y20G #current position to the 20th line
p #pasting
# join lines
J #join the next line with current line
# Search 
f* ; #find the character* in this line, ;find the next
/words n #type word and hit enter to search whole file, hit "n" for next in file
# Search - replace
:%s/find/replace/g #: "starts ex command"; 
                   # % specify first to last line or 1,5; 
                   # s - search & replace; 
                   # /find/replace/; 
                   # g - means global or only first case in each line
# edit multiple file
$vi file1 file2 file3 file4 #open multiple files
:n # next file :!n if quit save and go to next 
:N # previous file :!N if quit save and go to next 



# [3] Customizing Prompt
# p140 - 146



## ---------------------------- Common Tasks and Essential Tools ----------------------------- ##

# [1] Package Management - installing / maintaining software (package maintainer created package from the source code created by
#                                                             upstream provider, stored in central repo)
# <package system>        <Distribution>
# Debian style (.deb)     Debian, Ubuntu, Xandros, Linspire
# Red Hat style (.rpm)    Fedora, CentOS, Red Hat Linux, OpenSUSE, Mandriva, PCLinuxOS
#
# Packaging system tools - High level tool (Installing, removing, creating), low level tool (searching metadata, dependencies resolution)
# <Distributions>        <Low level Tool>         <High level Tool>
  Debian style           dpkg                     apt-get, aptitude
  Fedora,                rpm                      yum
  Red Hat Enterprise, 
  Linux, CentOS
# Find a packages in Repo
apt-get update
apt-cache search search_string # Debian
yum search search_string # Red Hat
# Install a package from a Repo
apt-get update
apt-get install package_name # Debian
yum install package_name # Red Hat
# Install a package from a package file (If the package file has been downloaded)
#                                        low level tool, if dependencies lack, error)
dpkg --install package_file # Debian
rmp -i package_file # Red Hat
# Removing a package
apt-get remove package_name # Debian
yum erase package_name # Red Hat
# Update a package from Repo
apt-get update
apt-get upgrade # Debian
yum update # red Hat
# Update a package from a package file
dpkg --install package_file
rpm -u package_file
# LIST all installed packages
dpkg --list # Debian
rpm -qa # Red Hat
# Determining whether a package is installed
dpkg --status package_name # Debian
rpm -q package_name # Red Hat
# Display information of a installed package
apt-cache show package_name # Debian
yum info package_name # Red Hat
# Finding which package installed a file
dpkg --search file_name # Debian
rpm -qf file_name # Red Hat



# [2] Storage media - manipulate data at devices level - manage storage devices
#                     (Not file level), physical storage, network storage, virtual storage
# P160 - P174
mount # Mount a filesystem
umount # unmount a filesystem
fdisk # Partition table manipulator
fsck # Check and repair a filesystem
fdformat # Format a floppy disk
mkfs # Create a filesystem
dd # Write block-oriented data directly to a device
genisoimage # Create an ISO 9660 image file
wodim # Write data to optical storage media
md5sum # Calculate an MD5 checksum



# [3] Networking 
# Check network information --
ping host_name # test whether its connected (IMCP ECHO_REQUEST)
#                "host_name" can be names or url, any host / CTRL-C to quit
traceroute xxxx.com # track and display all routers in order to get to specific local server system
netstat -ie # network interface of our network
        -r  # network's kernal / cofig information (Security)
# Move files across network --
# ftp (option 1)
ftp fileserver
# - ask for name:
# - ask for pass:
# log in promt of "ftp>" 
ls; cd # like unix file system to locate your file 
#      / move consol to the folder the file located
lcd /HOME/ubuntu # change local conslo to local dir -"local cd"
get xxxxxx.txt # transfer file on the remote folder to the local dir
bye # log off
# lftp (option 2) - more convient, popular option
lftp # ? Check more on internet
# wget (Option 3) - download file from both web and ftp sites
#        single file, multiple files, even whole site
wget http://xxxxxxx.com/xxxx.php # download first page
# check more detail in man page of wget
# scp (secure copy) / sftp (secure ftp) (option 4) - from "OpenSSh" package - leverage encrepty words
scp remote_system:xxxx.txt . # download a file from remote sys to the local dir
scp user@remote_system:xxxx.txt . # same but as "user" on remote sys
# pass:
sftp fileserver # same as "ftp", more secure
# - ask for name:
# - ask for pass:
# log in promt of "ftp>" 
ls; cd # like unix file system to locate your file 
#      / move consol to the folder the file located
lcd /HOME/ubuntu # change local conslo to local dir -"local cd"
get xxxxxx.txt # transfer file on the remote folder to the local dir
bye # log off
# Secure communication with remote hosts --
ssh # securely log in to a remote computer / in "OpenSSH" package / on port 22 on local
ssh remote_system # type "yes" to continue
ssh user@remote_system # login with different user on remote sys / type "yes" to continue
# password:
# login remote system$
exit # to log out



# [4] Searching for files
locate bin/zip # display all files match the strings (all sys)
find # search a given dir by the attributes of files to display
find ~ -type f -name "*.JPG" -size +1M | wc -l # define dir, attributes
find ~ \( -type f -not -perm 0600 \) -or \( -type d -not -perm 0700 \)
# dir, attributes, operator (-and, -or, -not, ()) logic operators
find ~ -type f -name "*.JPG" -size +1M -print # action
                                       -delete
                                       -ls
                                       -quit
find ~ -type f -and -name '*.BAK' -and -print # add "-and", easy to see logic
# user defined actions
-exec rm '{}' ';'
-ok rm '{}' ';' # ask yes? while exec
# -exec command current_path_name delimeter_of_end_command
find ~ -type f -name 'foo*' -ok rm '{}' ';'
# option - control the scope of "find"
-depth; -maxdepth levels; -mindepth levels; -mount; -noleaf
touch file # change times of file modification to current time
stat file # detail stat of a file
# Search REGEX
find dir -regex 'pattern' # match pathname
locate --regex 'pattern' # match pathname




# [5] Archiving / backup - timely backup system files; move large blocks of data
#                          file compression          ; place to plcae, devices to devices
# .gz = gzip / .bz2 = bzip2 / .tar = tar / .tgz = tar > gzip / .tbz = tar > bzip2 / .zip = zip
# compress file - gzip => xxx.gz
gzip file # compress to xxx.gz
gzip -c # write output to stdout, keep original
     -d # decompress, like gunzip
     -f # force compression, even on a compressed file
     -h # help manue
     -l # list compression stats for each file
     -r # recursively compress files
     -t # test intergity of a file
     -v # display long message while compressing
     -number # 1 - 9, fast - best compression, default - 6
gunzip file # restore back to xxx.txt
# compress file - bzip2 (differen arlgo as 'gzip', deep compress - 'demage', slow) => xxx.bz2
bzip2 file # compress to xxx.bz2
bzip2 -c # write output to stdout, keep original
      -d # decompress, like gunzip
      -f # force compression, even on a compressed file
      -h # help manue
      -l # list compression stats for each file
      -t # test intergity of a file
      -v # display long message while compressing
      -number # different as above
bunzip2 file # restore back to xxx.txt
bzip2recover file # recover demaged file
# *** DON'T compress compressed file - only becomes larger
# archiving files - gathering up multiple files, bundling them as backup (Copy dir)
#                 - also, move old data to somewhere as long-term storage
# tar [option 1] - archive
tar # Tape archiving utility
tar cf dir.tar dir # create an archive specifying name.tar, the dir should be just 'dir', 
#                                                           if ~/dir, all will be archived
    c # create an archive
    x # extract an archive
    r # Append / update an archive
    t # list content of an archive
tar tf dir.tar # list content of an archive
tar tvf dir.tar # List more detail of an archive
cd new.dir 
tar xf ../dir.tar # extract archive to the new location
# *** ownership will changed to performers
# tar [option 2] - create copy archive from local to USB and then extract to other machine
# check the device (mount automatically should) - usually, '/media/USB_name' = dir (can be sys, like '/home')
sudo tar cf /media/USB_name/dir.tar dir
umount # unmount a device ** check storage media
# plug in new machine ...
cd dir_master
sudo tar xf /media/USB_name/dir.tar # extract whole dir.tar
ls # check
# tar [option 3] - only use tar c/x/r/t on subset of a dir or files
sudo tar xf dir.tar '/.../...' # only extract specific dir or files
sudo tar xf dir.tar --wildcards '/.../.../*.txt' # Apply wildcard * in search
find dir -name 'file-A' -exec tar rf dir_sub.tar '{}' '+' # use find to match file, then update 'dir_sub' archive 
# tar [option 4] - find a subset dir or files, create an archive, compress the archive to .tgz or .tbz
find dir -name 'xxxx' | tar cf - --files-from=- | gzip > dir.tgz # use compress fun separately
find dir -name 'xxxx' | tar czf dir.tgz -T - # compress with gzip = czf
find dir -name 'xxxx' | tar cjf dir.tbz -T - # compress with bzip2 = cjf
# tar [option 5] - copy dir from a remote sys to local
mkdir remote_stuff_dir
cd remote-stuff_dir
ssh remote_sys 'tar cf - dir' | tar xf - # login in remote sys, create archive and show to stdout,
#                                          Extract archive in local from the stdout * '- stdout'
# pass:
ls # check
# zip - package and compress files *Linux user dont use much, only when exchange files with windows
# zip options zipfile file...
zip -r dir.zip dir # Create zip archive
#                    if -r 'recursive' not specify, only dir_name
#                    zip will automatically display loading message for each file
#                    2 OPTION: 'store' - no compression / 'deflate' - compress
cd new.dir 
unzip ../dir.zip # extract archive / if archive existed - only update new, promt ask replaced file
unzip -l ../dir.zip /.../... # Display files strcuture, not extract yet
      -v                     # Display more lists
# As 'tar', zip use stdin, stdout
find dir -name 'xxxx' | zip -@ xxxx.zip # use find to select, then create archive of subset files
#                                         -@ pipe a list of file names to zip
ls -l /dir/ | zip dir.zip - # zip takes stdin from other command, - stdinput
unzip -p dir.zip | less # unzip can pipe stdout to other command, -p pipe
# Synchronizing files and directories
rsync # synchronization | only update what is different | fast, economic
# rsync options source destination
# source & destination > (local file or dir, 
#                         remote file or dir in form '[user@]host:path',
#                         remote rsync server in form 'rsync://[user@]host[:port]/path')
rsync -av source.dir destination.dir # -a allow recursive archive, -v display infor
# [option 1 - sync the devices with local files]
mkdir /media/USB_name/backup # create backup file on devices
sudo rsync -av --delete dir1 dir2 dir3 /media/USB_name/backup # sync local to devices, 
#                                                               --delete remove any different files in devices
alias backup='sudo rsync -av --delete dir1 dir2 dir3 /media/USB_name/backup' # create an alias
# attach it to .bashrc file to initate backup when reboot, start
# [option 2 - sync the remote files with local files]
mkdir remote_sys_backup
sudo rsync -av --delete --rsh=ssh /remote_sys_backup remote-sys:/backup
# [option 3 - sync the local file with remote sync server]
mkdir remote_server
rsync -av -delete rsync://....edu/xxx/xxx/i386/os remote_server # URI - ex. resync.gtlib.gatech.edu


# [6] Regular Expressions - REGEX
grep # REGEX search through text --
# gerp [option] pattern [file...]
ls /../.. | grep zip # return lines that contain this string 'zip'
                 -i zip # ignore cases
                 -v zip # Return NOT match cases
                 -c zip # print count of matches, not actual lines
                 -l zip # print name of the files, not lines
                 -L zip # print name of the NOT match files, not lines
                 -n zip # Prefic each  matching line with the number of line
                 -h zip # for multifile search, suppress the output of filenames
grep -l bzip xxxx*.txt
# REGEX match pattern used --
^ $ . [] * {} - ?  + () | \
# basic #   # extend #
. <- grep '.zip' xxx*.txt # match any characters 
^ <- grep '^zip' xxx*.txt # match start with 'zip'
$ <- grep 'zip$' xxx*.txt # match end with 'zip'
[] <- grep '[bg]zip' xxx*.txt # match bzip or gzip - A set / select from it
           '[^bg]zip' xxx*.txt # match any string but bzip and gzip - in A set, ^ negation
           '[A-Z]zip' xxx*.txt # match Azip ... Zzip - in A set, range
           '[A-Za-z0-9]' xxx*.txt # match any characters, case, numbers
() <- grep '^([:upper:][:digit:])*' xxx*.txt # group strings into group and apply other meta
| <- grep 'AAA|^(BBB)|CCC' xxx*txt # 'or' to combine statement
? <- grep '\(?AAA\)?' xxx*.txt # match AAA or (AAA), string ? => string can show or not show, optional
\ <- grep '\(AAA\)' xxx*.txt # match (AAA), escape ()
* <- grep [[:upper:]] [[:upper:][:lower:]]*\. xxx*.txt # match a sentence, start with big case, 
                                                       # a lot words in between, end with'.', any number of times - 0 or more
                                                       # match - "This is a sentence."
+ <- grep [[:upper:]] [[:upper:][:lower:]]+\. xxx*.txt # Almost same as above, any number of times - at least 1 or more
{} <- grep -E '^\(?[0-9][0-9][0-9]\)? [0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$' xxx*.txt # match '(555) 432-2345'
{} <- grep -E '^\(?[0-9]{3}\)? [0-9]{3}-[0-9]{4}$' xxx*.txt # match '(555) 432-2345'
              '{n}' # occur exact n times
              '{n.m}' # at least n times, no more than m times
              '{n,}' # at least n times or more
              '{,M}' # no more than m times
# POSIX character classes
          [[:alnum:]] # any alphanumeric character
          [[:alpha:]] # any alphabetic character
          [[:digit:]] # any numeral
          [[:lower:]] # any lower case letter
          [[:upper:]] # any uppercase letter
# Search REGEX
find dir -regex 'pattern' # match pathname
locate --regex 'pattern' # match pathname
vim; less # type'/' then type pattern to search by regex (vim only support -basic regex-*)



# [7] Text Processing 
# concatenate file and print it on stdout
cat
cat file # print it
cat > file # type stdinput '>' as file content / CTRL-D to end
cat -ns file # print with blank line delimited & line numbers
# sort line by line of texts
sort 
sort > file # type stdinput '>' as file content (sort) / CTRL-D to end
sort file1 file2 file3 > file.all # meger all file sorted and into one file
sort -b file # ignore the leading space in each line
     -f # ignore case
     -u # remove duplicate cases from sorted result (GNU version only)
     -n # numeric sort
     -r # sort in reverse order
     -k # sort on a key field from field1 to field2 rather than entire line
     -m # treat each argument as a presorted file, merge multiple file into a single sorted result
     -o # send sorted output to a file rather than stdout
     -t # define field separater, default - tab, spaces
ls -l /usr/bin | sort -nr -k 5 | head # -nr -> sort by numeric, reverse, in filed 5 -> size
sort -k 1,1 -k 2n file # sort field1 only by character, for field2 use numeric sort
sort -k 3.7nbr -k 3.1nbr -k 3.4nbr # sort field3 7th character
                                   # sort filed3 1st character
                                   # sort filed3 4th character
sort -t ':' -k 7 file # sort file delimted by ':' by the field7
# Report or Omit Repeated lines
uniq # usually use with 'sort'
sort file1 | uniq > file2 # since uniq only deduplicate adjacent like aabbcc
                          # abcabc doesn't work
uniq -c file # a list of duplicated lines with number of freqency 
     -d file # output only repeated lines
     -f n file # ignore n leading fields in each line, by white space
     -i file # ignore case
     -s n file # skip the leading n characters of each line
     -u file # output only unique lines
# Remove sections from each line of files
cut 
cut -c n-m file # extract n or n-m characters 
    -f n file # extract nth filed
       n,m,h file
    -d ':' # set delemlieter :
    --complement # extract entire line of text, except portions specified by -c / -f
cut -f 3 file # extract 3rd field
cut -f 3 file | cut -c 7-10 # extract 3rd filed and 7-10 characters
cut -d ':' -f 1 file | head # set delimeter as :, 1st field, list top 10
# Merge lines of files (By columns)
paste file1 file2 > file.all # combine columns in file1 with columns in file2
# Join lines of files (Like SQL)
join file1 file2 > file.all # need proimary key in all file to join, sorted. Find more on 'man' page
# Compare two sorted files line by line
comm 
comm file1 file2 # 'only file1, only file2, overlap'
comm -12 file1 file2 # -n suppress columns
# compare files line by line
diff 
diff -c file1 file2 # (none)-no diff, - first file, + seconf file, ! a line chnaged
# *** file1 2011-22-23 00:00:00.00000000000 -05000
# --- file2 2011-22-23 00:00:00.00000000000 -05000
# **************
# *** 1,4 ***
# - a
#   b
#   c
#   d
# --- 1,4 ---
#   b
#   c
#   d
# + e
diff -u file1 file2 # (none)-no diff, - first file, + second file
# --- file1 2011-22-23 00:00:00.00000000000 -05000
# +++ file2 2011-22-23 00:00:00.00000000000 -05000
# @@ -1,4 +1,4 @@
# -a
#  b
#  c
#  d
# +e
# Apply diff 'updates' to the original file (Previous version)
patch # it accept output from 'diff' to convert older file to newer file
# find more options on 'man' page
diff -Naur old_file new_file > diff_file # Create diff file -> files
diff -Naur old_dir new_dir > diff_file # Create diff file -> dirs
patch < diff_file # patching file1 with file2 version
# file1 => file2
# Transliterate or Delete characters
tr # convert a set of characters to a corresponding set of characters
echo "lower cases letter" | tr a-z A-Z # transliterate a-z to A-Z
"LOWER CASES LETTER"
echo "lower cases letter" | tr [:lower:] A # transliterate a lower class to 'A'
"AAAAA AAAAA AAAAAA"
# a set of values can be:
ABCDEFGHIJKLMNOPQRST # a list
A-Z # a range
[:upper:] # a POSIX class
# decoding - ROT13
echo "secrt text" | tr a-zA-Z n-za-mN-ZA-M # encoding
"frperg grkg"
echo "frperg grkg" | tr a-zA-Z n-za-mN-ZA-M # decoding
"secrt text"
# squezzing text
echo "aaaabbbbccc" | tr -s ab # -s -> eliminate repeated and adjacent letters
"abccc"
# Stream Editor for filtering and transforming text by line---------***
sed # perform text editing on a stream of text - through a command or 
    # a script file of multiple commands to perform editing
# s/search/replacement/ - 'search can be regex'
echo "front" | sed 's/front/back/' # search everyline to find 'front' and replace with 'back'
"back"
echo "front" | sed '2s/front/back/' # only perform on line 2, yet, no line 2, no editing
"front"
sed -n '1,5p' file # print file from line 1 to line 5
sed -n '$p' file # print last line
sed -n '/regex/p' file # print match regex
sed -n '/regex/!p' file # DON'T print match regex
sed -n '1~2' file # start at line 1 take 2 step (lines) (including 1 - ex. 1,2) as interval to the line 3, every odd line
sed -n '4,+5' file # strat at line 4 and then 5 lines down included, total 6 lines
sed 's/\([0-9]\{2\}\)\/\([0-9]\{2\}\)\/\([0-9]\{4\}\)$/\3-\1-\2/' file
# to replace '11/25/2008' with '2008-11-25'
sed 's/regexp/replacement/' file
'regexp' -> [0-9]{2}/[0-9]{2}/[0-9]{4}$ # 11/25/2008, Q position is different from 2008-11-25
([0-9]{2})/([0-9]{2})/([0-9]{4})$ # use () to create sub-expression
'replacement' -> \3-\1-\2 # 1,2,3 represent the position of each sub-expression, rearrange the orders, use '-', use '\' to escape 1,2,3
 # add to 's/regexp/replacement/', then use '\' to escape all () {} which may confuse 'sed'
 echo "aaabbbccc" | sed 's/b/B/' # only replace one letter (leading) per line
 "aaaBbbccc"
 echo "aaabbbccc" | sed 's/b/B/g' # global replacement
 "aaaBBBccc"
 sed -i 's/xxx/ddd/' file # directly edit file, not print 
 sed 's/xxx/ddd/; s/fff/ddd/' file # with multiple editing 
 # take a sed file to perform editing
 sed -f script.sed file 
 script.sed
 # ----script.sed----- #
 # sed script to produce Linux distributions report
 1 i\
 \
 Linux Distributions Report\

 s/\([0-9]\{2\}\)\/\([0-9]\{2\}\)\/\([0-9]\{4\}\)$/\3-\1-\2/
 y/abcdefghijklmnopqrst/ABCDEFGHIJKLMNOPQRST/
 # --------end-------- #
 # Check text spelling
 aspell check file # will display option, choose to replace wrong spelling
 aspell -H check file # HTML file, make it ignore tag text

# [8] Formatting output

# [9] Printing 

# [10] Compiling Programs


## ---------------------------- Writing Shell Scripts ----------------------------- ##
# [1] Writing first scripts
# step1-Write a script / 
> mark.sh
vi mark.sh
# ----- mark.sh ----- #
#!/bin/bash                     -> shebang, indicating 'bash' interpreter location

# this is our script            -> Description of the script can do

echo "Hello World!"           # -> Scripts / try to use long option --all --directory (readability)

find dir \
   \( \
   	    -type f \
   	    -not -perm 0600 \
   	    -exec chmod 0600 '{}' ';' \   # Indention / Line continuation for readability
   \) \
   -or \
   \( \
   	    -type d \
   	    -not -perm 0700 \
   	    -exec chmod 0700 '{}' ';' \
   \)
# ----- end ---------- #
# step2-Make the script executable / 
ls -l mark.sh # check execuation premission -rw-r--r--
chmod 775 mark.sh # change execution premission to executable for everyone, 700 only executable for owner
ls -l mark.sh # check again
# step3-Put the script somewhere the shell can find it
/home/ubuntu/mark.sh # specify full path, otherwise ssh can't find it
bash mark.sh # or use 'bash' to directly specify interpreter 'bash'
# in order to directly execute program
echo $PATH # display 'PATH' var which is a ':' separated search directories
# cause most 'PATH' var contains '/home/me/bin' to allow user execute their own command
# if NOT contain -
export PATH=~/bin:"$PATH" # add this command to .bashrc file
source .bashrc # reload the source file, reload
# create local 'bin'
cd
mkdir bin
mv mark.sh bin
mark.sh # directly execute this sommand


# [2] starting a project
# create variable in script
TITLE="This is a title." # a string
TITLE2=TITLE # another variable
TITLE3="we have $TITLE2" # a string contains variable
TITLE4=(ls -l xxx.txt) # a command
TITLE5=$((7*8/90)) # an arithmetic expansion
TITLE6="\t\ta string\n" # escape sequence such as tabs, newlines 
$(command) # directly use the stdout of a command
echo $TITLE # if 'TITLE' not exists, system will create a 'blank' one
echo "<HTML>
          <HEAD>
                 <TITLE>$TITLE2</TITLE>
          </HEAD>
      <HTML>" # use echo to output text
# ----- end ------ #
# Here document / Here script
command <<- token # --------------------- #
            text  # as input line by line #
            token # --------------------- #
                  # any comand takes 'stdinput' can be used
# ------ script example <1> -------- #
#!/bin/bash

# Script to extract ftp file

FTP_SERVER=ftp.nl.debian.org # define variables
FTP_PATH=/debian/dist/...../.../
REMOTE_FILE=xxxx.txt

ftp -n <<- _EOF_ # token starts
       open $FTP_SERVER
       user anonymous me@linuxbox
       cd $FTP_PATH
       hash
       get $REMOTE_FILE
       bye
       _EOF_ # token ends

ls -l $REMOTE_FILE # check result
# ------------ end ---------------#
# ------ script example <2> -------- #
#!/bin/bash

# Script to display sys infor

TITLE="system Information Report For $HOSTNAME"
CURRENT_TIME=$(date +"%x %r %z")
TIME_STAMP="Generated $CURRENT_TIME, by $USER"

cat <<- _EOF_
<HTML>
          <HEAD>
                 <TITLE>$TITLE</TITLE>
          </HEAD>
          <BODY> 
                 <H1>$TITLE</H1>
                 <p>$TIME_STAMP</p>
          </BODY>
<HTML>
_EOF_
# ------------ end ---------------#



# [3] Top Down Design
# design top level tasks first, then define subtasks for each
# <1>main task
#    (1)suntask
#    (2)subtask
# <2>main task
#    (1)suntask
#    (2)subtask
# shell function
function_name () {
  local.variable # local var which only exists during fun session
                 # can share name with global var, no impact
  command
  command
  return # terminate fun session
}
# Debugging with 'stub'
function_name () {
  echo "this fun executed." # Return information regrading execution of each part
  return
}
script.file.sh # run scrpit to see if everypart executable (Debugging)
# ------ script example -------- #
#!/bin/bash

# Script to display sys infor

TITLE="system Information Report For $HOSTNAME" # create global bariables
CURRENT_TIME=$(date +"%x %r %z")
TIME_STAMP="Generated $CURRENT_TIME, by $USER"

function_name1 () { # define shell functions 
  cat <<- _EOF_
          <H2>xxxx</H2>
          <PRE>$(command1)</PRE>
          _EOF_
  return
}
function_name2 () {
  cat <<- _EOF_
          <H2>xxxx</H2>
          <PRE>$(command2)</PRE>
          _EOF_
  return
}
function_name3 () {
  cat <<- _EOF_
          <H2>xxxx</H2>
          <PRE>$(command3)</PRE>
          _EOF_
  return
}

cat <<- _EOF_ # actuall commands start to execute in this script
<HTML>
          <HEAD>
                 <TITLE>$TITLE</TITLE> # lookup global variables
          </HEAD>
          <BODY> 
                 <H1>$TITLE</H1>
                 <p>$TIME_STAMP</p>
                 $(function_name1) # lookup shell functions
                 $(function_name2)
                 $(function_name3)
          </BODY>
<HTML>
_EOF_
# ------------ end ---------------#



# [4] Flow control - IF statements
[] # old test, can't escape (),...
[[]] # new test, no worry escape
# pseudocode (use 'if' 'else' to translate computer language)
if [[command]]; then
	  do_this # can be another if-else loop
elif [[command]]; then
	  do_that # can be another if-else loop
else 
	  do_it # can be another if-else loop
fi
# test statement
[[statement]] # perform a test, returns 'true' or 'false'
# basic if-else example
x=5

if [[ "$x" = 5 ]]; then # "$x" - warp by "" defend the error caused by missing value x, "" always returns a string
	   echo "x equals 5."
elif [[ "$x" = 7 ]]; then
	   echo "x equals 7."
elif [[ "$x" = 8 ]]; then
	   echo "x equals 8."
else
	   echo "x does not equal to any."
fi
# exit status
if [[]] # returns 1 or 0, if 0 execute
0 - 255
0 # success
1 - 255 # others - unsuccess
#ex.
ls -d /usr/bin
echo $?
0 # command executed successfully
true # default returns 0
false # default returns 1
#
# USING Testing - file, strings, integer
[['test']]
# FILE testing
fil1 -ef file2 # two file names refers to the same file by hard linking
file1 -nt file2 # file1 is newer than file2
file1 -ot file2 # file1 is older than file2
-b file # file exists, a block-special (devices) file
-c file # file exists, a character-special (devices) file
-d file # file exists, is a directory
-e file # file exists
-f file # file exists, is a regular file
-g file # file exists, is set-group-ID
-G file # file exists, is owned by the effective group ID
-k file # file exists, has its "sticky bit" set
-L file # file exists, is a symbolic link
-O file # file exists, is owned by the effective user ID
-p file # file exists, is a named pipe
-r file # file exists, is readable
-s file # file exists, has a length greater than zero
-S file # file exists, is a network socket
-t file_descripter # whether standrad in/out/error is being redirected 
-u file # file exists, is setuid
-w file # file exists, is writable
-x file # file exists, is executable
# -------------example------------------ #
test_file () {

	# test file: xxxxxxxxxxxxxxxxxxxxx

	FILE=~/.bashrc

	if [ -e "$FILE" ]; then
		    if [ -f "$FILE" ]; then
		    	    echo "xxxxxxxxxxxxxx1"
		    fi
		    if [ -d "$FILE" ]; then
		    	    echo "xxxxxxxxxxxxxx2"
		    fi
		    if [ -r "$FILE" ]; then
		    	    echo "xxxxxxxxxxxxxx3"
		    fi
	else 
		    echo "xxxxxxxxxxxany"
    fi
  return
}
# --------------- end ------------------- #
#
# STRINGS testing
string # string is not NULL
-n string # the length of the string is greater than zero
-z string # the length of the string is zero
string1 == string2 # two strings are equal
string1 != string2 # ........... are not equal
string1 > string2 # string1 sorts after string2
string1 < string2 # string1 sorts before string2
# -------------example------------------ #
#!/bin/bash

# test-string: evaluate the vale of a string

ANSWER=maybe

if [ -z "$ANSWER" ]; then
	     echo "xxxxxxxxxxx1"
	     exit 1
fi
if [ "ANSWER" == "yes" ]; then
	     echo "xxxxxxxxxxx2"
elif [ "$ANSWER" == "no" ]; then
	     echo "xxxxxxxxxxx3"
elif [ "$ANSWER" == "maybe" ]; then
	     echo "xxxxxxxxxxx4"
else
	     echo "xxxxxxxxxxxother"
fi
# --------------- end ------------------- #
#
# Integer testing
integer1 -eq integer2 # equal
integer1 -ne integer2 # not equal
integer1 -le integer2 # leess or equal
integer1 -lt integer2 # less
integer1 -ge integer2 # greater or equal
integer1 -gt integer2 # greater
# -------------example------------------ #
#!/bin/bash

# test-integer: evaluate the vale of a integer

INT=-5

if [ -z "$INT" ]; then
	    echo "INT is empty."
	    exit 1
fi

if [ "$INT" -eq 0 ]; then
	 echo "INT is Zero."
else
	 if [ "$INT" -lt 0 ]; then
	 	   echo "xxxxxxxxxxxnegative"
	 else
	 	   echo "xxxxxxxxxxxpositive"
	 fi
     if [ "$((INT % 2))" -eq 0 ]; then
	 	   echo "xxxxxxxxxxxeven"
	 else
	 	   echo "xxxxxxxxxxxodd"
	 fi
fi
# --------------- end ------------------- #
#
# Match regex
[["$strings" =~ regex]]
#ex.
if [[ "$INT" =~ ^-?[0-9]+$ ]]; then # if it is an integer
if [[ "$FILE" == foo.* ]]; then # match all file 'foo.*', enable expansion
#
# Arithmetic truth test
((100 * 8 / 50)) # supports full arithmetic evaulation
(( INT == 0))
(( INT < 0))
# also, test
if ((1)); then echo "true"; fi # return true if integer is non-zero
if ((0)); then echo "true"; fi
#
# Combining Expression
Operation     test     [[]] and (())
AND           -a       &&
OR            -o       ||
NOT           !        !
# example
if [[ INT -ge MIN_VAL && INT -le MAX_VAL ]]; then # and
if [ "$INT" -ge MIN_VAL -a "$INT" -le MAX_VAL ]; then # and
if [[ !(INT -ge MIN_VAL && INT -le MAX_VAL) ]]; then # NOT ( AND )
if [ ! \("$INT" -ge MIN_VAL -a "$INT" -le MAX_VAL\) ]; then # NOT ( AND ) *need escape []
#
# Control Operators
&& ||
command1 && command2 # execute command2, only if command1 executed successfully
command1 || command2 # execute command2, only if command1 executed unsuccessfully
#ex.
mkdir temp && cd temp # if successfully make a dir 'temp', then cd to that dir
[ -d temp] || mkdir temp # if dir 'temp' not exists (False test), make a dir called 'temp'
#
# More examples - recode previous function
# ------------- example ------------------ #
report_home_space () {
	if [[ $(id -u) -eq 0 ]]; then
		   cat <<- _EOF_
		           .......
		           .......
		           _EOF_
    else
    	   cat <<- _EOF_
		           .......
		           .......
		           _EOF_
    fi
    return
}
# --------------- end -------------------- #






# [5] Reading keybroad input into the program (User interactive)
read [-option] [variable...] # read a single line of standrd input
read -a array # assign input to array with index zero
     -d delimiter # the first charater in 'delimiter' is to indicate end of input
     -e # use readline to handle input, one line one var
     -n num # read a number of characters in input rather than all
     -p prompt # Display aprompt for input
     -r # rawmodw, not inperpret backlash characters as escape
     -s # Don't echo characters as they are typed (For password)
     -t seconds # time out, if a non-zero exit return
     -u fd # use input from a file descriptor fd, rather than standard input
# single var
echo -n "Please enter a number -> " # n, no new line
read INT
echo "var = '$INT'"
# multiple vars
echo -n "Please enter multiple numbers -> "
read INT1 INT2 INT3 INT4 INT5 INT6 # if less - empty, if more, all extra for the last var
echo "var1 = '$INT1'"
echo "var2 = '$INT2'"
echo "var3 = '$INT3'"
echo "var4 = '$INT4'"
echo "var5 = '$INT5'"
echo "var6 = '$INT6'"
# if no var assign
echo -n "Please enter multiple numbers -> "
read
echo "var = '$REPLY'" # if input "a c d f c", REPLY = "a c d f c"
                      # REPLY, shell var for store 'read' input if no var assign
# use IFS to separate input
IFS = ":" read var1 var2 var3 var4 <<< "$FILE" # use ":" as separators to read the value of a file into 
                                               # different vars
                                               # use a variable assignment right in front of a command
                                               # IFS = ":" in front of read, IFS old value restore after 
                                               # end of this 'read' command
                                               # <<< redirect value of FILE as standard input for 'read'
# use 'menu' for user to choose
echo "
Please select:

1. xxxxxxxxxxxxxx
2. xxxxxxxxxxxxxx
3. xxxxxxxxxxxxxx
0. quit
"
read -p "Enter selection [0-3] > "

# Validating input
# -- define error fun
invalid_input() {
	echo "Invalid input '$REPLY'" >&2
	exit 1
}
# -- set up test
[[ -z $REPLY ]] && invalid_input # Is it empty? && -> if 1 success, ex 2
(( $(echo $REPLY | wc -w) > 1 )) && invalid_input # is it multiple items? && -> if 1 success, ex 2
# example.ex
# ------------- example ------------------ #
#!/bin/bash

# test-integer: evaluate the vale of a integer

echo -n "Please enter a number -> " # -> ask for single input
read INT

invalid_input() {
	echo "Invalid input '$INT'" >&2 # -- define error fun
	exit 1
}

[[ -z $INT ]] && invalid_input # Is it empty? 
(( $(echo $INT | wc -w) > 1 )) && invalid_input # is it multiple items? 


if [ -z "$INT" ]; then # start function
	    echo "INT is empty."
	    exit 1
fi

if [ "$INT" -eq 0 ]; then
	 echo "INT is Zero."
else
	 if [ "$INT" -lt 0 ]; then
	 	   echo "xxxxxxxxxxxnegative"
	 else
	 	   echo "xxxxxxxxxxxpositive"
	 fi
     if [ "$((INT % 2))" -eq 0 ]; then
	 	   echo "xxxxxxxxxxxeven"
	 else
	 	   echo "xxxxxxxxxxxodd"
	 fi
fi
# --------------- end -------------------- #



# [6] Flow control: Looping with 'WHILE' and 'UNTIL'
# WHILE - if not meet stoping requirement, repeat process
while [commands]; do # if meet [] returns exit status 1
       commands 
done
# -- define WHILE Loop
# ------------- example ------------------ #
#!/bin/bash

# Define stoping function

DELAY=3 # define times to delay for showing result

while [[ $REPLY != 0 ]]; do # define the 0 as quit, test on it
    clear
	cat "xxxxxxxxxx, 0 quit"
	read -p "Please select > "
   if [[ $REPLY = ...0123 ]]; then
	       if [[ $REPLY = ...1 ]]; then
		      command
		      sleep $DELAY # delay showing
	       fi
	       if [[ $REPLY = ...2 ]]; then
		      command
		      sleep $DELAY
	       fi
           if [[ $REPLY = ...3 ]]; then
		      command
		      sleep $DELAY
	       fi
	else 
	   echo "Invalid input"
	   sleep $DELAY
	fi
done
echo "Program terminated." # if user type 0, other wise repeat the loop
# --------------- end -------------------- #

# -- endless loop while, break with "break"
# ------------- example ------------------ #
#!/bin/bash

# Define 'break' in the loop

DELAY=3 # define times to delay for showing result

while true; do # true always return exit 0, loop run endlessly
    clear
	cat "xxxxxxxxxx, 0 quit"
	read -p "Please select > "
   if [[ $REPLY = ...0123 ]]; then
	       if [[ $REPLY = ...1 ]]; then
		      command
		      sleep $DELAY # delay showing
		      continue # more efficient execution, continue will skip over other, no reason to test other
	       fi
	       if [[ $REPLY = ...2 ]]; then
		      command
		      sleep $DELAY
		      continue
	       fi
           if [[ $REPLY = ...3 ]]; then
		      command
		      sleep $DELAY
		      continue
	       fi
	       if [[ $REPLY = ...0 ]]; then
		      break # define 'break' in loop, other wise endless loop
	       fi
	else 
	   echo "Invalid input"
	   sleep $DELAY
	fi
done
echo "Program terminated." # if user type 0, other wise repeat the loop
# --------------- end -------------------- #

# until - opposite of while, exit when status = 0
until [commands]; do # if meet [] returns exit status 0
       commands 
done
# -- keep trying, looping until it success = status 0
# ------------- example ------------------ #
#!/bin/bash

# until-count: display a series of numbers

count=1

until [ $count -gt 5 ]; do # count > 5?, count = 1 which < 5, ststua 1, looping
	  echo $count
	  count=$((count + 1))
done
echo "Finished."
# --------------- end -------------------- #

# -- Reading file with loop
# ------------- example ------------------ #
#!/bin/bash

# read file with loop, exit when all lines been read

# example1
while read var1 var2 var3; do
	  command...
done < xxxxx.txt # > redirect 'xxxx.txt' as stdinput to supply loop while

# example2
sort -k 1,1 -k 2n xxxxx.txt | while read var1 var2 var3; do # | supply the result of prev as stdin
                                                            # vars in command will gone after execution
                                                            # | execute loop command in subshell
	  command...
done
# --------------- end -------------------- #



# [7] Trouble Shooting (Debugging Technics)


# [8] Flow Control: Branching with CASE
case $var in
	pattern1)  command
               ;;
    pattern2)  command
               ;;
    pattern3)  command
               ;;
esac
# pattern example
a) # matches if equals a
[[:alpha:]]) # matches a single alphabetic character
???) # matches exactly three characters long
*.txt) # matches ends with '.txt'
*) # match any value -> used as 'else'
# multiple patterns
pattern1 | pattern2)



# [9] Positional Parameters


# [10] # Flow Control: Looping With 'FOR'
# original form
for i in A B C D; do # a list "A B C D"
	  echo $i;
done

for i in {A..D}; do # a brace expandsion {}
	  echo $i;
done

for i in lala*.txt; do # a pathname expandsion *.txt
	  echo $i;
done
# C language form
for (( i=0; i<5; i=i+1 )); do # ((exp1; exp2; exp3)) -> exp1 initiate, exp2 determin finish?, exp3 each iteration
	  echo $i
done


# [10] Strings and Numbers






## <<<< Hadoop HDFS commandline >>>> ##














