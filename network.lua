--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 11/8/16
-- Time: 01:55
--

require 'nn'
require('MaskConv.lua')
--local utils = paths.dofile'utils.lua'


local function createModel(opt)
   print(opt)
   local function testModel(model)
--      model:double()
      local imageSize = 28
      local input = torch.randn(1,1,imageSize,imageSize):type(model._type)
--      print('it works', input)
      print('forward output',{model:forward(input)})
--      print('output', model.output)
      print('backward output',{model:backward(input,model.output)})
      model:reset()
   end

   local dim = 16 -- TODO: refactor
   local model = nn.Sequential()
   model:add(nn.Reshape(1, 28, 28))
   model:add(nn.MaskConv('nn', 'A', 1, dim, 7, 7, 1, 1, 3, 3))
   model:add(nn.MaskConv('nn', 'B', dim, dim, 3, 3, 1, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(dim, dim, 1, 1))
   if opt.crit == 'softmax' then
      model:add(nn.SpatialConvolution(dim, 256, 1, 1))
      model:add(nn.Reshape(256, 1, 28, 28))
--      model:add(nn.View(-1):setNumInputDims(2))
   else
      model:add(nn.View(-1):setNumInputDims(3))
      model:add(nn.Linear(28*28*dim, 28*28))
      model:add(nn.Sigmoid())
   end

   testModel(model)

   -- init
   for k,v in pairs(model:findModules'nn.Linear') do
      v.bias:zero()
   end

   for k,v in pairs(model:findModules('nn.MaskConv')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end

   return model
end

return createModel