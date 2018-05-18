import argparse
import numpy as np
import caffe
from caffe.proto import caffe_pb2

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with dataset')
    parser.add_argument('--train-proto', default='../model/mobilenet_v2_deploy.prototxt', help='Path to deploy prototxt')
    parser.add_argument('--prefix-model', default='../model/', type=str, help='Path to pretrained weights')
    #parser.add_argument('--test-proto', default=, help='Path to test proto')
    parser.add_argument('--solver', default='../model/solver.prototxt', help='Solver path')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3], help='List of device ids')
    parser.add_argument("--timing", action='store_true', help="Show timing info.")

    args = parser.parse_args()
    return args

def define_Solver(args):

    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE
    s.train_net = args.train_proto
    s.test_net.append(args.train_proto)
    s.test_interval = 500
    s.test_iter.append(100)
    s.max_iter = 10000
    s.type = "SGD"
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 1000
    s.snapshot = 5000
    s.snapshot_prefix = args.prefix_model
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    with open(args.solver, 'w') as f:
      f.write(str(s))
'''
def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))
'''

#def solve(proto, snapshot, gpus, timing, uid, rank):


def train_net(args):


    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(args.train_proto, phase=caffe.TRAIN)
    net.forward()
    embedding=net.blobs['embedding'].data
    
    define_Solver(args)

    solver = None
    solver = caffe.get_solver(args.solver)

    niter = 1
    test_interval = 1
    for it in range(args.niter):
        solver.step(solver.param.max_iter)
        if it % test_interval == 0:
            print('Iteration %s test', it)

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__=='__main__':
    main()
