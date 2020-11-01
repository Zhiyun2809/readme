#GITHUB 
#username zhiyun2809
#email zhiyun.chen@live.de
#password Emmy2009%
# get info
$git config --global user.name
$git config --global user.email

#source
#https://rubygarage.org/blog/most-basic-git-commands-with-examples

#Configuration
$ git config --global user.name "Zhiyun2809"
$ git config --global user.email "zhiyun.chen@live.de"
# turn on code highlighting
$ git config --global color.ui true
# get list
$ git config --list

#=====================================================
# Add files
$ git add my_new_file.txt
$ git add .
# Add files recursive
$ git add --all
$ git add -A

#=====================================================
# remove file from reporsitory, untrack file
$ git rm --cached my-files.txt
$ git reset my-files.txt

#=====================================================
# Commit existing files
$ git commit -m "Add new files"
# Add new files and commit
$ git commit -a -m "Add some files and commit all"
# undo last commit, and move HEAD by one commit
$ git reset --soft HEAD
# alternative
$ git add file-i-forgot-to-add.html
$ git commit --amend -m "add the remaining file"
# individual commit for file a and file b
$git add file_a
$git commit file_a -m "bugfix, in a"
$git add file_b
$git commit file_b -m "bugfix, in b"
# commit all existing file
$git add --all
$git commit -a

#=====================================================
# Push To and Pull From a remote repository 
# push local to remote repository 
# get domainname as origin
$ git remote add origin https://github.com/Zhiyun2809/mysite.git
# push 
$ git push -u origin master
$ git push
# get remote status
$ git remote -v


#=====================================================
# branch
#=====================================================
# get branch asterisk marks the current branche i am in
$ git branch
# create new branch
$ git branch new-branch
# switch to new-branch
$ git checkout new-branch
# combine above 2 steps
$ git checkout -b new-branch-name
# push the new branch remotely
$ git push --set-upstream origin new-branch-name
$ git push

# 

# get data from remore server
$ git pull

# rename a branch
$ git branch -m old-branch-name new-branch-name
# alternative
$ git branch --move old-branch-name new-branch-name

# merge branch to master
$ git checkout master
$ git merge new-branch
$ git push

# delete branch
$ git branch -D branch-name
# alternative
$ git branch --delete branch-to-delete

#show all branches
$ git branch -a

# chekcout remote branch-name as new-branch-name
$git checkout -b new-branch-name origin/branch-name
# checkout remote branch and keep tracking
$ git checkout --track origin/branch-name #check out branch

# set up new branch mimic orign on server 
$ git push --set-upstream origin new-branch

# push new branch to remote repo
git push -u https://github.com/Zhiyun2809/roche01.git new-branch-name
# push to master
git push -u https://github.com/Zhiyun2809/roche01.git master

#=====================================================
# get log info
#=====================================================
$git help log
$git log 
$git log --all
# view the most recent 3 commits
$git log -3
# compare
$git diff
# filter by author
$git log --author <name>
$git log --commitor <name>
$git log --author=Bob
# filter by date
$git log --before <yyyy-mm-dd>
$git log --after <yyyy-mm-dd>ranch 
$git log --after <yyyy-mm-dd> --before <yyyy-mm-dd>

$git log --graph

#=====================================================
# remove file from filesystem and repository
$git rm file1.txt
$git commit -m "remove file1.txt"
# remove only from he Git repository but not from the filesystem
$git rm --cached file1.txt
$git commit -m "remove file1.txt"

# get remove file
$ rm file.py
$ git rm file.py


#=====================================================

#=====================================================
# workflow
$ git init
$ git remote add origin https://github.com/Zhiyun2809/roche01.git
$ git remote -v
$ git fetch origin
$ git checkout --tractk origin/dry #check out branch
$ git checkout --track origin/master #check out mastert

# chekcout remote 
$ git checkout -b new-branch-name origin/branch

$ git branch new-branch # make local new-branch
$ git checkout new-branch # switch to new-branch
$ git add --all     #add changes to new-branch
$ git commit -m "new-branch"
$ git push --set-upstream origin new-branch # push new-branch to remote repo
#push changes to remove new-branch
# origin as remote 
$ git push -u origin new-branch     #push changes to remote new-branch

# push changes to remote repo
$ git push origin branch_name

# refresh/pull renew information from remote repo
$ git pull 
#=====================================================
# workflow

# retrieve single file from repo
$ git checkout <branch-name> -- <filename>

$ git checkout <filename>

# alternative
$ git checkout branch-name
# list all local branches 
$ git branch 
# list all remote branches 
$ git branch -a

$ git checkout master

#=====================================================
# get log information
$ git log
<Ctrl+Z>


#=====================================================
# retrieve 
$ git reset HEAD file.py
$ git checkout file.py

#=====================================================
.gitignore
foldername

#=====================================================
# clone
#=====================================================
# clone remote master repository in the local folder demo
$ mkdir demo
$ cd demo
$ git clone https://......git

# downlow all branches from remote repository add .git at the end
$ git clone --mirror https://.. git .git
# 2nd step
$ git config --bool core.bare false
$ git reset --hard

t cl# clone remote to local
$ git clone -b new-branch-name origin/remove-branch-name
# remove all log info, d
$ rm -dfr .git
# init brand new git 
$ git init
$ git add .
$ git commit -m "new project"


# clone to other folder
$ git clone https://github.com/Zhiyun2809/mysite.git another-name
# if above command not work, try below one
$ git clone https://github.com/Zhiyun2809/mysite.git another-name
# pull
# get latest version
$ git pull

# compare between different branch
$ git diff <branch1> <branch2>
