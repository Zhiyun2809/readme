from google.colab import drive
drive.mount('')


!pip install -q keras


!pip freeze


jupyter notebook
get def argument shift+tab

%%time

%% bsash
ls
pwd
whoami

# python tips
# print large number
print(f'{total:,}')

name = 'chen'
daughter = 'emmy'
print(f'{daughter} is {name} daugher')

# context manager
with open ('test.txt','r') as f:
    file_contenx = f.read()

# loop over two list
names_list = []
for name, hero in zip(names_list, heros_list, more_list):
    print(f'{name} is {hero}')

# unpack tubple
my_tuple = (1,2)
a, b  = my_tuple
a, _ = my_tuple
my_tuple = (1,2,3,4,5,6)
a, b, *c  = my_tuple
a, b, *_  = my_tuple
a, b, *c, d = my_tubple


# command line 
username = input('Username: ')
passwork = input('Password: ')

print('Logging In ...')





