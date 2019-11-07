# 410246-Laboratory-Practice-I-410243-Data-Analytics-
410241:: High Performance Computing<br>
410242:: Artificial Intelligence and Robotics <br>
410243:: Data Analytics <br>

For Hadoop Word Count Assignment: <br>
Make sure the file is executable:<br>

 chmod +x reducer.py <br>

Testing <br>
Make sur your two programs work. Here's a simple series of test you can run: <br>
  cat mapper.py | mapper.py <br>
This will make mapper.py output all the words that make up its code.<br>
 cat mapper.py | mapper.py | sort | reducer.py <br>
This will generate the (unsorted) frequencies of all the unique words (punctuated or not) in mapper.py. <br>
Running on the Hadoop Cluster<br>
Let's run the Python code on the Ulysses.txt file. <br>
We'll assume that the Python code is stored in ~hadoop/352/dft/python <br>
We'll assume that the streaming java library is in ~hadoop/contrib/streaming/streaming-0.19.2-streaming.jar <br>
We'll also assume that ulysses.txt is in dft and that we want the output in dft-output:<br>
cd<br>
cd 352/dft/python<br>
hadoop dfs -rmr dft1-output<br> 
hadoop jar /home/hadoop/hadoop/contrib/streaming/hadoop-0.19.2-streaming.jar -file ./mapper.py \ <br>
        -mapper ./mapper.py -file ./reducer.py -reducer ./reducer.py  -input dft -output dft-output<br>
Changing the number of Reducers<br>
To change the number of reducers, simply add this switch -jobconf mapred.reduce.tasks=16 to the command line:<br>
cd<br>
cd 352/dft/python <br>
hadoop dfs -rmr dft1-output <br> 
hadoop jar /home/hadoop/hadoop/contrib/streaming/hadoop-0.19.2-streaming.jar \ <br>
        -jobconf mapred.reduce.tasks=16 \ <br>
        -file ./mapper.py \ <br>
        -mapper ./mapper.py \ <br>
        -file ./reducer.py \ <br>
        -reducer ./reducer.py \  <br>
        -input dft -output dft-output <br>
