@echo off
javac -d bin src/dataShuffle/Shuffle.java
cd bin
java dataShuffle.Shuffle
pause