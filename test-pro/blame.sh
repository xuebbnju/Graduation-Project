echo off
cd "E:/graduate-design/git-project"
for file in `ls -a` 
do 
   cd "E:/graduate-design/git-project/"$file
   git ls-files | grep -E "\.java$" | sed 's/\(.*\.\)java$/git blame \1java > \1blame/' | sh 
   echo $file 完毕
   cd ..
done
