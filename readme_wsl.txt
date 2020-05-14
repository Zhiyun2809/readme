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
