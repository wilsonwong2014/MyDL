# ========================================
#       simple cnn with tensorboard
# ========================================
# define conv kernel
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    weight = tf.Variable(initial_value=initial)
    return weight

# define conv bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    bias = tf.Variable(initial_value=initial)
    return bias

# define a simple conv operation 
def conv_op(in_tensor, kernel, strides=[1,1,1,1], padding='SAME'):
    conv_out = tf.nn.conv2d(in_tensor, kernel, strides=strides, padding=padding)
    return conv_out

# define max pooling operation
def max_pool_2x2(in_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
    max_pool = tf.nn.max_pool(in_tensor, ksize, strides, padding)
    return max_pool       
def simple_cnn_tensorboard(mnist):
    '''
    simple cnn with tensorboard visulization

    '''
    tf.reset_default_graph()
    log_dir = './simple_cnn_log' # log dir to store all data and graph structure
    sess = tf.InteractiveSession()

    # cnn structure
    max_steps = 1000
    learning_rate = 0.001
    dropout = 0.9
    w1 = [5,5,1,32]
    b1 = [32]
    w2 = [5,5,32,64]
    b2 = [64]
    wfc1 = [7*7*64,1024]
    bfc1 = [1024]
    wfc2 = [1024,10]
    bfc2 = [10]

    def variable_summaries(name,var):
        with tf.name_scope(name+'_summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar(name+'_mean', mean)
        with tf.name_scope(name+'_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar(name+'_stddev', stddev)
        tf.summary.scalar(name+'_max',tf.reduce_max(var))
        tf.summary.scalar(name+'_min',tf.reduce_min(var))
        tf.summary.histogram(name+'_histogram', var)

    with tf.name_scope('input'):    
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        keep_prob = tf.placeholder(tf.float32,name='kp')

    with tf.name_scope('image_reshape'):
        x_image = tf.reshape(x,[-1, 28, 28, 1]) # 28*28 pic of 1 channel
        tf.summary.image('input', x_image)

    # 1st layer    
    with tf.name_scope('conv_layr1'):
        W_conv1 = weight_variable(w1);    variable_summaries('w1',W_conv1)
        b_conv1 = bias_variable(b1);      variable_summaries('b1',b_conv1)
        with tf.name_scope('Wx_plus_b'):
            pre_act = conv_op(x_image, W_conv1)+b_conv1
            tf.summary.histogram('pre_act',pre_act)
        h_conv1 = tf.nn.relu(pre_act, name='activiation')       
        h_pool1 = max_pool_2x2(h_conv1)

    # 2nd layer
    with tf.name_scope('conv_layr2'):
        W_conv2 = weight_variable(w2);    variable_summaries('w2',W_conv2)
        b_conv2 = bias_variable(b2);      variable_summaries('b2',b_conv2)
        h_conv2 = tf.nn.relu(conv_op(h_pool1, W_conv2)+b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    # fc1
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        W_fc1 = weight_variable(wfc1);    variable_summaries('w_fc1',W_fc1)
        b_fc1 = bias_variable(bfc1);      variable_summaries('b_fc1',b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1, name='_act')
        # drop out
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

    # fc2
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable(wfc2);    variable_summaries('w_fc2',W_fc2)
        b_fc2 = bias_variable(bfc2);      variable_summaries('b_fc2',b_fc2)
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2,name='fc2_softmax')
        #tf.summary.scalar('softmax', y_conv)

    # loss function
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]), name='cross_entropy')
        #tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):    
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # estimate accuarcy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.arg_max(y_conv,1), tf.arg_max(y_,1))
        accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #acc_summary = tf.summary.scalar('accuarcy', accuarcy)
    tf.summary.scalar('accuarcy', accuarcy)

    # summary all 
    merged = tf.summary.merge_all()
    #merged = tf.summary.merge([input_summary,acc_summary])
    train_writer = tf.summary.FileWriter(log_dir+'/train',sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+'/test')

    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x:xs, y_:ys, keep_prob:k}

    saver = tf.train.Saver()

    for i in range(max_steps):
        if i%10 == 0:
            summary, acc = sess.run([merged, accuarcy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s'%(i, acc))
        else:
            if i%100 == 99:
                continue
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                      options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%03d'%i)
                train_writer.add_summary(summary,i)
                saver.save(sess,log_dir+'/model.ckpt',i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)                
    train_writer.close()
    test_writer.close()  

    return

