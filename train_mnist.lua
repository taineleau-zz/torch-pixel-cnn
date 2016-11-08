--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 11/7/16
-- Time: 13:46
--

-- load torchnet:
local tnt = require 'torchnet'
require 'cudnn'

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-crit', 'sigmoid', 'use sigmoid or softmax')
local config = cmd:parse(arg)
print(config)
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

local net = require('network')(config) --nn.Sequential():add(nn.Linear(784,10))
print(net)
local criterion =  cudnn.VolumetricCrossEntropyCriterion() -- nn.MultiLabelMultiClassCriterion()--nn.BCECriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
engine.hooks.onStartEpoch = function(state)
--   meter:reset()
--   clerr:reset()
end
engine.hooks.onSample = function(state)
--   print('here!!!', state.sample.input)
   state.sample.input:div(256)
   if config.crit == 'softmax' then
      local len = math.floor(math.sqrt(state.sample.input:size(2)))
      state.sample.target:resize(state.sample.input:size(1), 1, len, len)
   else
      state.sample.target:div(256)
   end
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
      if config.crit == 'softmax' then
      igpu:resize(state.sample.input:size()):copy(state.sample.input:div(256))
      local len = math.floor(math.sqrt(state.sample.input:size(2)))
      tgpu:resize(state.sample.input:size(1), 1, len, len)
           :copy(state.sample.target)
      else
         igpu:resize(state.sample.input:size()):copy(state.sample.input:div(256))
         tgpu:resize(state.sample.output:size()):copy(state.sample.target:div(256))
      end
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

-- generate net:
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}

