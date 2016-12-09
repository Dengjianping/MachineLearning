# remenber use chmoe +x script.sh to make it executabel
echo "Input something"
read variable 
echo "Hello, ${variable}"

# readonly variable just cannot be changed
variable = "jdeng"
readonly variable
echo ${variable}

# delete a variable
variable = "kk"
unset variable

# $$ 表示当前Shell进程的ID，即pid, 对于 Shell 脚本，就是这些脚本所在的进程ID。
echo $$

# 当前脚本的文件名
echo $0

# 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。例如，第一个参数是$1，第二个参数是$2
echo $n

# 传递给脚本或函数的参数个数。
echo $#
# 传递给脚本或函数的所有参数。
echo $*
# 传递给脚本或函数的所有参数。被双引号(" ")包含时，与 $* 稍有不同
# 当它们被双引号(" ")包含时，"$*" 会将所有的参数作为一个整体，
# 以"$1 $2 … $n"的形式输出所有参数；"$@" 会将各个参数分开，以"$1" "$2" … "$n" 的形式输出所有参数。
echo $@
# 上个命令的退出状态，或函数的返回值。
echo $?

# loop
for i in $@
do
    echo "$i"
done

for i in $*
do
    echo "$i"
done

# array
a = (1,2,3,4)

# show all elements
${a[*]}
${a[@]}
# get length of array
length = ${#a[*]}

# control flow
a=30
b=40
if [$a==$b]
then
    echo "$a"
fi

# case structure
a= 40
case #a in
    1) echo "$a"
    ;;
    40) echo "$a"
    ;;
    *) echo "all"
    
# deine a function
function Hello() {
    echo "$1"
    echo "$2"
    echo "$3"
}

Hello 1 2 3

# output redirect
# this will be covered
23 > example.txt

# this will be appended
23 >> example.txt

# input redirect

# algorithm
a=2
b=4
c=`expr $a + $b`
echo $c