Roche sudo password: same as windows 
#update system
sudo apt update
sudo apt upgrade

# install c compile
sudo apt-get install gcc

# install ImageMagick
sudo apt-get install imagemagick

# manual install imagemagick
wget http://www.imagemagick.org/download/ImageMagick.tar.gz
tar xvzf ImageMagick.tar.gz
cd ImageMagick-*
./configure
make
sudo make install
sudo ldconfig /usr/local/lib

# ==============================================================
# linux command

#!/bin/bash

# mmv
# rename all files starting with letter "a" to "b"
mmv a\* b\#1
mmv '*abc*' '#1xyz#2'
# print out output instead actually rename it
mmv -n a\* b\#1
man mmv

m find all files with extension in subfolders
find . -type f -name *.tif
# case sensitive
find . -type f -iname *.tif
# one depth
find . -maxdepth 1 -type f -iname *.jpg
# delete 
find . -type f -name *.tif -delete
# find file by date
find . -type f -ls |grep 'May 20'

# delete files older than 30 days
find /folder -type f -mtime +30 -exec rm -f {} \
find /folder -name "*.jpg" -type f -mtime +30 -exec rm -f {} \

# move files by date
for i in `ls -lrt |grep 'May 12'|awk '{print $9}'`; do cp $i /new_dest; done
for i in `ls -lrt */* |grep 'May 12'|awk '{print $9}'`; do cp $i new_dest/; done


# list all files with only digits in name
ls -l |grep -Eo '[[:digits:]]*.jpg'
# alternative
ls -l |grep -E '[0-9]{1,3}'.jpg			[0-9] with {1,3} digits number

# extract info from jpg
identify -verbose 12.jpg |grep -E "frame|^x|^y"

# for i in `ls |grep -E '[0-9]{1,3}'.jpg |awk '{print $i}'`; do identify
-verbose $i |grep -E 'frame|^x|^y' |tr '\n' ','; done

# remove whitespace use tr
tr -d '[:space:]'

# zip
# test validity of zip file
untip -tq [filename].zip

# unzip all file to subfolder with same name
for i in `ls *.zip`; do unzip "$i"; done

# zip all subfolder
for i in `ls `; do zip -r $i.zip $i; done

# go to subfolder exec command
for i in ./3030*/;
do
(cd "$i" && echo 'hello'>hello.txt);
done

#  list number of files
ls |wc -l

#  list all subfolders
ls -l |grep '^d'
ls -d */

# -zip all subfolder into individual zip file
for i in `ls -l |grep '^d'|awk '{print $9}'`; do zip -r $i.zip $i; done
