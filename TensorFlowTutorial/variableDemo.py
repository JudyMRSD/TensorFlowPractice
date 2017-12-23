import tensorflow as tf   # import the tensorflow module and call it tf

x = tf.constant(35, name='x')  #create a constant value x, and give it the numerical value 35
y = tf.Variable(x + 5, name='y')   # create a variable called y, and define it as being the equation x+5
#w = tf.Variable(<initial-value>, name=<optional-name>)
z = tf.Variable(10) # create a variable called z, and define an initial value 10
model = tf.global_variables_initializer()  #return an Op that initializes global variables
with tf.Session() as session:     #create a session for computing the values
    # run the Op to initialize values
    #print(session.run(z)) #value in z not initialize to the initialize value 10 yet, so run(z) returns error
    session.run(model)
    #run calculation to compute the value for y
    print(session.run(y))
    print(session.run(z))
