--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 11/7/16
-- Time: 13:46
--

-- load torchnet:
local tnt = require 'torchnet'

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function()

         -- load MNIST dataset:
         local mnist = require 'mnist'
         local dataset = mnist[mode .. 'dataset']()
         dataset.data = dataset.data:reshape(dataset.data:size(1),
            dataset.data:size(2) * dataset.data:size(3)):double()

         -- return batches of data:
         return tnt.BatchDataset{
            batchsize = 128,
            dataset = tnt.ListDataset{  -- replace this by your own dataset
               list = torch.range(1, dataset.data:size(1)):long(),
               load = function(idx)
                  return {
                     input  = dataset.data[idx],
                     target = dataset.data[idx], -- unsupervised learning
                     --target = torch.LongTensor{dataset.label[idx] + 1},
                  }  -- sample contains input and target
               end,
            }
         }
      end,
   }
end

-- set up logistic regressor:
local net = dofile('network.lua')() --nn.Sequential():add(nn.Linear(784,10))
print(net)
local criterion = nn.BCECriterion()--nn.MultiLabelSoftMarginCriterion()--nn.ClassNLLCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
engine.hooks.onStartEpoch = function(state)
--   meter:reset()
--   clerr:reset()
end
engine.hooks.onSample = function(state)
--   print('here!!!', state.sample.input)
   state.sample.input:div(256)
   state.sample.target:div(256)
end

engine.hooks.onForwardCriterion = function(state)
--   print('target:', state.sample.target:nElement())
--   print('output:', state.network.output:nElement())
--   meter:add(state.criterion.output)
--   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print('loss:', state.criterion.output)
--      print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
--         meter:value(), clerr:value{k = 1}))
   end
end

-- set up GPU training:
if config.usegpu then

   -- copy model to GPU:
   require 'cunn'
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input:div(256))
      tgpu:resize(state.sample.target:size()):copy(state.sample.target:div(256))
      state.sample.input  = igpu
      state.sample.target = tgpu
--      print(igpu)
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   lr        = 0.2,
   maxepoch  = 5,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}
print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))

