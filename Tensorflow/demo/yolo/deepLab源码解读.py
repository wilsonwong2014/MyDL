#deeplabV3+源码分解学习
#   https://www.jianshu.com/p/d0cc35b3f100
#   https://github.com/tensorflow/models/tree/master/research/deeplab
'''
    github上deeplabV3+的源码是基于tensorflow（slim）简化的代码，是一款非常值得学习的标准框架结构。
基于这份代码，可以学习到很多关于如何标准化构建大型深度学习网络的相关编写知识。
'''

#=======================================================================
#一，dataset 读取(关于dataset的写入生成我们放在后面，这里假设数据准备好了）
#tensorflow已经不流行用原始的数据读取的方法，而是用slim更加简单方便。但是这里要看懂还是需要tensorflow数据读取的那一块基本知识
slim = tf.contrib.slim
dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder
#准备数据集的feature
  keys_to_features = {      
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
#准备数据集的执行句柄
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }
#合并为解码器
  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)
#得到整合后的dataset
  dataset= dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
#得到dataset的生成器，应该是可以从dataset中获取东西
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
#利用get函数可以获取得到需要的东西
  image, height, width = data_provider.get(
      [common.IMAGE, common.HEIGHT, common.WIDTH])
  label, = data_provider.get([common.LABELS_CLASS])
#对image 和 label进行预处理
#比如 random_scale,random_crop,flip_dim(img,label)，有底层代码
#整合为sample
  sample = {
      common.IMAGE: image,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width
  }
#得到训练用到的一个一个batch
samples=tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)
#设置slim的优先队列
inputs_queue = slim.prefetch_queue.prefetch_queue(
          samples, capacity=128 * config.num_clones)
#在网络开始获取一个数据包
samples = inputs_queue.dequeue()
#添加名字，后期才可以summary
samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=common.IMAGE)
samples[common.LABEL] = tf.identity(
      samples[common.LABEL], name=common.LABEL)


#=======================================================================
#二，网络细节

#=======================================================================
#二/1，image_pyramid 这里可能有多尺度变换的成分
for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_width, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)
      if model_options.crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])
#然后求一个reduce_max

#=======================================================================
#二/2，Encoder阶段 (extract_features)
#2.1 给定预先训练网络抽取feature输入：image 输出：end_points,concat_logits
#这里在模型选择和构建的写法上很有意思，我们这里不做衍生。关键看这里怎么复用训练好的模型来搞事情。
#用的是Xception
#xception的源码比较简单，而是是开源的，可以直接拿来用，也有预先训练好的模型，最主要要注意的应该就是命名。
#他这里主要是把卷积变成了空洞卷积和分离卷积，其他没有任何改变。

#=======================================================================
#2.2 aspp 模块合成feature
branch_logits = []
#pooling 层
pool_height = scale_dimension(model_options.crop_size[0],
                                        1. / model_options.output_stride)
pool_width = scale_dimension(model_options.crop_size[1],
                                       1. / model_options.output_stride)
image_feature = slim.avg_pool2d(
              features, [pool_height, pool_width], [pool_height, pool_width],
              padding='VALID')
image_feature = slim.conv2d(
              image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
image_feature = tf.image.resize_bilinear(
              image_feature, [pool_height, pool_width], align_corners=True)
image_feature.set_shape([None, pool_height, pool_width, depth])
branch_logits.append(image_feature)
#1*1 卷积
branch_logits.append(slim.conv2d(features, depth, 1,
                                         scope=ASPP_SCOPE + str(0)))
#按照上图的稀疏卷积
for i, rate in enumerate(model_options.atrous_rates, 1):
    scope = ASPP_SCOPE + str(i)
     if model_options.aspp_with_separable_conv:
              aspp_features = split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            branch_logits.append(aspp_features)
        concat_logits = tf.concat(branch_logits, 3)
合并这些层，输出
concat_logits = slim.conv2d(
            concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
concat_logits = slim.dropout(
            concat_logits,
            keep_prob=0.9,
            is_training=is_training,
            scope=CONCAT_PROJECTION_SCOPE + '_dropout')

#=======================================================================
          decoder_features = features
          for i, name in enumerate(feature_list):#just to scope
            decoder_features_list = [decoder_features,gradient_res]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
#对endpoint进行1*1卷积
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
#然后对于单层resize
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
#每次resize完都要重新定义
              decoder_features_list[j].set_shape(
                  [None, decoder_height, decoder_width, None])
            decoder_depth = 256
            if decoder_use_separable_conv:
#然后合并卷积
              decoder_features = split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')

#=====================================================================
#3，Decoder阶段(refine_by_decoder)输入：features，end_points 输出：features（gradient为我修改部分）
          decoder_features = features
          for i, name in enumerate(feature_list):#just to scope
            decoder_features_list = [decoder_features,gradient_res]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
#对endpoint进行1*1卷积
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
#然后对于单层resize
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
#每次resize完都要重新定义
              decoder_features_list[j].set_shape(
                  [None, decoder_height, decoder_width, None])
            decoder_depth = 256
            if decoder_use_separable_conv:
#然后合并卷积
              decoder_features = split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')

#========================================================================
#3，多GPU并行计算
#这里主要用了model_deploy这个类来完成model的clone，运用的是并行GPU运算的方法。几个GPU同时计算，然后算出一个平均结果，作为最后的参数。
#第一步，配置参数
  config = model_deploy.DeploymentConfig(#a class for muti-GPUs
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)
#第二步 配置模型和模型参数
  model_fn = _build_deeplab
  model_args = (inputs_queue, {
          common.OUTPUT_TYPE: dataset.num_classes
      }, dataset.ignore_label)
clones = model_deploy.create_clones(config, model_fn, args=model_args)
#第三步 计算总体优化结果
total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)#get the gradient and loss in total
#在这个类里面有很多底层的模型分类，这里暂时不做介绍。

#==============================================
#4，summary
#设置summary主要是为了调试时候查看数值变化时候用的。如何加入也是一种十分重要的方法
#1，新建summaries
summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
#2，添加所有变量
for model_var in slim.get_model_variables():#all the model variables
      summaries.add(tf.summary.histogram(model_var.op.name, model_var)) 
#3，可能会添加输入输出的图片信息
summary_image = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
summaries.add(
          tf.summary.image('samples/%s' % common.IMAGE, summary_image))
      summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
summaries.add(
          tf.summary.image('samples/%s' % common.LABEL, summary_label))
summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
      summaries.add(
          tf.summary.image(
              'samples/%s' % common.OUTPUT_TYPE, summary_predictions))
#4，添加loss
for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
#5，添加直方图
for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))
#6，合并所有summary
    summary_op = tf.summary.merge(list(summaries))


#===========================================
#5，update
#基本的训练套路
#添加优化方式
optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
grad_updates = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)#
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)#* is used to split
#软着陆
session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

# 开始训练
    slim.learning.train(
    train_tensor,
    logdir=FLAGS.train_logdir,yiiqi
    log_every_n_steps=FLAGS.log_steps,
    master=FLAGS.master,
    number_of_steps=FLAGS.training_number_of_steps,
    is_chief=(FLAGS.task == 0),
    session_config=session_config,
    startup_delay_steps=startup_delay_steps,
    init_fn=train_utils.get_model_init_fn(
    FLAGS.train_logdir,
    FLAGS.tf_initial_checkpoint,
    FLAGS.initialize_last_layer,
    last_layers,
    ignore_missing_vars=True),
    summary_op=summary_op,
    save_summaries_secs=FLAGS.save_summaries_secs,
    save_interval_secs=FLAGS.save_interval_secs)

6，eval
# 定义评估标准
    metric_map = {}
    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))
#开始评估
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)

7，save
#保存整个模型
from tensorflow.python.tools import freeze_graph
    saver = tf.train.Saver(tf.model_variables())
    tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
    freeze_graph.freeze_graph_with_def_protos(
        tf.get_default_graph().as_graph_def(add_shapes=True),
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        _OUTPUT_NAME,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)

8，read
读取deeplab模型，freeze_graph的形式
class DeepLabModel(object): 
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    # Input name of the exported model.
    _INPUT_NAME = 'ImageTensor:0'

    # Output name of the exported model.
    _OUTPUT_NAME = 'SemanticPredictions:0'


    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'


    def __init__(self, INPUT_PATH):
        """
        Creates and loads pretrained deeplab model.
        """
        
        self.graph = tf.Graph()

        graph_def = None
        with gfile.FastGFile(INPUT_PATH+self.FROZEN_GRAPH_NAME+'.pb','rb') as f:
            graph_def=tf.GraphDef.FromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            self.sess = tf.Session(graph=self.graph,config=config)
        
        """
        config = tf.ConfigProto(allow_soft_placement=True) 
        config.gpu_options.allow_growth = True
        
        ckpt=tf.train.get_checkpoint_state(INPUT_PATH)
        new_saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        self.sess=tf.Session(config=config)
        new_saver.restore(self.sess,ckpt.model_checkpoint_path)
        self.graph=tf.get_default_graph()
        """

    def run(self, image):
        """
        Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        _input=self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
        _out=self.graph.get_tensor_by_name(self.OUTPUT_TENSOR_NAME)
        batch_seg_map = self.sess.run(
        _out,
        feed_dict={_input: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

MODEL=DeepLabModel(INPUT_PATH)
resized_im,seg_map=MODEL.run(original_im)
mask_save=change_to_3_channels(seg_map)
seg_map_show=Image.fromarray(mask_save.astype(np.uint8))

9，数据生成
这里主要用到ImageReader这个类，最主要是记住tf.train.Example(features=tf.train.Features{})这个类

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='jpeg', channels=3):
    """Class constructor.

    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Image channels.
    """
    with tf.Graph().as_default():
      self._decode_data = tf.placeholder(dtype=tf.string)
      self._image_format = image_format
      self._session = tf.Session()
      if self._image_format in ('jpeg', 'jpg'):
        self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
      elif self._image_format == 'png':
        self._decode = tf.image.decode_png(self._decode_data,
                                           channels=channels)

  def read_image_dims(self, image_data):
    """Reads the image dimensions.

    Args:
      image_data: string of image data.

    Returns:
      image_height and image_width.
    """
    image = self.decode_image(image_data)
    return image.shape[:2]

  def decode_image(self, image_data):
    """Decodes the image data string.

    Args:
      image_data: string of image data.

    Returns:
      Decoded image data.

    Raises:
      ValueError: Value of image channels not supported.
    """
    image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
    if len(image.shape) != 3 or image.shape[2] not in (1, 3):
      raise ValueError('The image channels not supported.')

    return image


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))



def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
  """Converts one image/segmentation pair to tf example.

  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.

  Returns:
    tf example of one image/segmentation pair.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(
          _IMAGE_FORMAT_MAP['jpg']),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'image/segmentation/class/encoded': (
          _bytes_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature(
          'png'),
  }))

def image_seg_to_tfexample_Gradient(image_data, filename, height, width, seg_data,gradient):
  """Converts one image/segmentation pair to tf example.

  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.

  Returns:
    tf example of one image/segmentation pair.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/gradient':_int64_list_feature(gradient),
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(
          _IMAGE_FORMAT_MAP['jpg']),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'image/segmentation/class/encoded': (
          _bytes_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature(
          'png'),
  }))

test_num_images=len(test_list)
test_num_per_shard=int(math.ceil(test_num_images/float(_NUM_SHARDS)))
test_image_reader=build_data_HTF.ImageReader('jpeg',channels=3)
test_label_reader=build_data_HTF.ImageReader('png',channels=1)

for shard_id in range(_NUM_SHARDS):
    output_filename=os.path.join(output_path,'test-%05d-of-%05d.tfrecord'%(shard_id,_NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * test_num_per_shard
        end_idx = min((shard_id + 1) * test_num_per_shard, test_num_images)
        for i in range(start_idx,end_idx):
            image_file_name=os.path.join(test_org_path,test_list[i]+'.jpg')
            img_cv=cv2.imread(image_file_name,0)
            img_gradient=cv2.Laplacian(img_cv,cv2.CV_64F)
            img_gradient=np.astype(img_gradient,np.int64)
            image_data=tf.gfile.FastGFile(image_file_name,'rb').read()
            height,width=test_image_reader.read_image_dims(image_data)
            seg_file_name=os.path.join(test_seg_path,test_list[i]+'.png')
            seg_data=tf.gfile.FastGFile(seg_file_name,'rb').read()
            seg_height,seg_width=test_label_reader.read_image_dims(seg_data)
            if height!=seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')
            example=build_data_HTF.image_seg_to_tfexample_Gradient(image_data,test_list[i],height,width,seg_data,img_gradient)
            tfrecord_writer.write(example.SerializeToString())
            print("%d / %d finished"%(i,end_idx-start_idx))

作者：horsetif
链接：https://www.jianshu.com/p/d0cc35b3f100
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

