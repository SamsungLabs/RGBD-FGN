import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Dropout

from . import utils

''' Modified based on "https://github.com/charlesq34/pointnet2" '''
class Pointnet2(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self._build()
        
    def _build(self):
        # Set Abstraction Layers
        self.sa1 = PointnetSA(npoint=1024, radius=0.04, nsample=64, mlp=[32,32,64], mlp2=None, group_all=False, bn=True)
        self.sa2 = PointnetSA(npoint=256, radius=0.08, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, bn=True)
        self.sa3 = PointnetSA(npoint=64, radius=0.16, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, bn=True)
        self.sa4 = PointnetSA(npoint=16, radius=0.32, nsample=64, mlp=[256,256,512], mlp2=None, group_all=False, bn=True)

        self.sa_layers = []
        self.sa_layers.append(self.sa1)
        self.sa_layers.append(self.sa2)
        self.sa_layers.append(self.sa3)
        self.sa_layers.append(self.sa4)

        # Feature Propagation Layers
        self.fp1 = PointnetFP(mlp=[128,128], bn=True)
        self.fp2 = PointnetFP(mlp=[128,128], bn=True)
        self.fp3 = PointnetFP(mlp=[128,64], bn=True)
        self.fp4 = PointnetFP(mlp=[64,64], bn=True, last=True)
          
        self.fp_layers = []
        self.fp_layers.append(self.fp1)
        self.fp_layers.append(self.fp2)
        self.fp_layers.append(self.fp3)
        self.fp_layers.append(self.fp4)

    def call(self, inputs, training=False):
        l0_xyz = tf.slice(inputs, [0,0,0], [-1,-1,3])
        l0_points = None
        
        xyz_dict, points_dict = dict(), dict()
        xyz_dict[0] = l0_xyz
        points_dict[0] = l0_points

        for i, sa_layer in enumerate(self.sa_layers):
            xyz_dict[i+1], points_dict[i+1]  \
                = sa_layer(xyz_dict[i], points_dict[i], training) 
        
        n = len(self.sa_layers)
        for i, fp_layer in enumerate(self.fp_layers):
            points_dict[n-1-i] \
                = fp_layer(xyz_dict[n-1-i], xyz_dict[n-i],
                                points_dict[n-1-i], points_dict[n-i], training)

        return dict(xyz=xyz_dict,
                    features=points_dict)

########################################################################################
################################ PointNet2 Layer #######################################
########################################################################################
class Conv2d(Layer):
    def __init__(self, filters, strides=[1, 1], padding='VALID', activation=tf.nn.relu, bn=False, dp=False):
        super(Conv2d, self).__init__()
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.dp = dp
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape = (1, 1, input_shape[-1], self.filters),
            initializer = 'glorot_normal',
            trainable = True,
            name='conv2d'
        )
        
        if self.bn:
            self.bn_layer = BatchNormalization()
        if self.dp:
            self.dp_layer = Dropout(rate=0.3)

        super(Conv2d, self).build(input_shape)

    def call(self, inputs, is_training=False):
        outputs = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn:
            outputs = self.bn_layer(outputs, training=is_training)
        if self.activation:
            outputs = self.activation(outputs)
        if self.dp:
            outputs = self.dp_layer(outputs, training=is_training)
        
        return outputs


class PointnetSA(Layer):
    def __init__(self, npoint, radius, nsample, mlp, mlp2, group_all, bn=False, dp=False, pooling='max', knn=False, use_xyz=True):
        super(PointnetSA, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.mlp2 = mlp2
        self.group_all = group_all
        self.bn = bn
        self.dp = dp
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.mlp_list = []
        self.mlp2_list = []

    def build(self, input_shape):
        for i, filters in enumerate(self.mlp):
            self.mlp_list.append(Conv2d(filters=filters,bn=self.bn,dp=self.dp))
        if self.mlp2 is not None:
            for i, filters in enumerate(self.mlp2):
                self.mlp2_list.append(Conv2d(filters=filters,bn=self.bn,dp=self.dp))
        super(PointnetSA, self).build(input_shape)
    
    def call(self, xyz, points, is_training=False):
        if self.group_all:
            new_xyz, new_points, _ = utils.sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, _ = utils.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.knn, self.use_xyz)
        
        for i, mlp_layer in enumerate(self.mlp_list):
            new_points = mlp_layer(new_points, is_training=is_training)

        if self.pooling=='max':
            new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True, name='maxpool')
        elif self.pooling=='avg':
            new_points = tf.math.reduce_mean(new_points, axis=2, keepdims=True, name='avgpool')
        elif self.pooling=='max_and_avg':
            max_points = tf.math.reduce_max(new_points, axis=2, keepdims=True, name='maxpool')
            avg_points = tf.math.reduce_mean(new_points, axis=2, keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        if self.mlp2 is not None:
            for i, mlp_layer in enumerate(self.mlp2_list):
                new_points = mlp_layer(new_points, is_training=is_training)

        return new_xyz, tf.squeeze(new_points, 2)


class PointnetSAMSG(Layer):
    def __init__(self, npoint, radius_list, nsample_list, mlp, bn=False, dp=False, use_xyz=True):
        super(PointnetSAMSG, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp = mlp
        self.bn = bn
        self.dp = dp
        self.use_xyz = use_xyz
        self.mlp_list = []

    def build(self, input_shape):
        for i in range(len(self.radius_list)):
            mlp_list_tmp = []
            for j, filters in enumerate(self.mlp[i]):
                mlp_list_tmp.append(Conv2d(filters=filters,bn=self.bn,dp=self.dp))
            self.mlp_list.append(mlp_list_tmp)
        super(PointnetSAMSG, self).build(input_shape)
    
    def call(self, xyz, points, is_training=False):
        new_idx = utils.farthest_point_sample(self.npoint, xyz)
        new_xyz = utils.gather_point(xyz, new_idx)
        new_points_list = []
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = utils.query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = utils.group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = utils.group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            for j,mlp_layer in enumerate(self.mlp_list[i]):
                grouped_points = mlp_layer(grouped_points, is_training=is_training)
            new_points = tf.math.reduce_max(grouped_points, axis=2)
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)

        return new_xyz, new_points_concat, new_idx


class PointnetFP(Layer):
    def __init__(self, mlp, bn=False, dp=False, last=False):
        super(PointnetFP, self).__init__()
        self.mlp = mlp
        self.bn = bn
        self.dp = dp
        self.last = last
        self.mlp_list = []

    def build(self, input_shape):
        for i, filters in enumerate(self.mlp):
            if i == len(self.mlp) - 1:
                if self.last:
                    self.mlp_list.append(Conv2d(filters=filters,activation=None,bn=False,dp=False))
                else:
                    self.mlp_list.append(Conv2d(filters=filters,bn=self.bn,dp=self.dp))
            else:
                self.mlp_list.append(Conv2d(filters=filters,bn=self.bn,dp=self.dp))
        super(PointnetFP, self).build(input_shape)
    
    def call(self, xyz1, xyz2, points1, points2, is_training=False):
        dist, idx = utils.three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.math.reduce_sum((1.0/dist),axis=2,keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = utils.three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        
        for i, mlp_layer in enumerate(self.mlp_list):
            new_points1 = mlp_layer(new_points1, is_training=is_training)
        new_points1 = tf.squeeze(new_points1, 2) # B,ndataset1,mlp[-1]

        return new_points1