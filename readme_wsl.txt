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

# linux look for file with date
ls -l
 -rw-rw-rw-    1    root     system          943   Aug   09   02:59  File

for j in `ls -l |awk '{ if ($7 == "09") print $9}'`
    do
        mv $j $Destination;
    done

# copy file last 30 days
fine . -mtime -30 -exec cp {} targetdir \;

# bash head
#!/bin/bash


# alternative
fine . -mtime -90 -ls >tmp.txt
for f in `cat tmp.txt`;
do cp $f /new_folder/
done

# list subfolder
ls -d
