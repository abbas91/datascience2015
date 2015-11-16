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





































## <<<< Hadoop HDFS commandline >>>> ##














