--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 11/8/16
-- Time: 02:01
--

local MaskConv, Parent = torch.class('nn.MaskConv', 'nn.Module')

function MaskConv:__init(data_type, mask_type,
                         nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,
                         verbose)
   -- data_type: `cudnn` or `nn`
   -- mask_type: `A` or 'B' or nil (none)
   -- this module is masked before weight updates,
   -- technically, we apply a mask to `gradOutput` before `accGradParameters()`

   Parent.__init(self)

   if mask_type then
      self.m = torch.Tensor(kH, kW):fill(1)
      if ((kH + 1) / 2 < kH) then
         self.m[{{(kH + 1)/2, kH}, {(kW + 1)/2 + 1, kW}}] = 0
         self.m[{{(kH + 1)/2 + 1, kH}, {1, kW}}] = 0
      end
      if mask_type == 'A' then
         self.m[(kH + 1)/2][(kW + 1)/2] = 0
      end
   end
   if self.verbose then
      print('mask', self.m)
   end
   self.mask = torch.Tensor(nOutputPlane, nInputPlane, kH, kW) -- not use forloop in lua to avoid slowdown
   for i = 1, nOutputPlane do
      for j = 1, nInputPlane do
         self.mask[i][j] = self.m
      end
   end

   if data_type == 'cudnn' then
   else
      self.conv = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   end

   local w, gw = self.conv:parameters()
   self.weight = w[1]
   self.bias = w[2]
   self.kH = kH
   self.kW = kW
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.verbose = verbose

end

function MaskConv:updateOutput(input)
   self:clearWeight()
   self.output = self.conv:forward(input)
   return self.output
end

function MaskConv:updateGradInput(input, gradOutput)
   self.gradInput = self.conv:updateGradInput(input, gradOutput)
   return self.gradInput
end

function MaskConv:accGradParameters(input, gradOutput, scale)
   self.conv:accGradParameters(input, gradOutput, scale)
end

function MaskConv:clearWeight()
   if self.verbose then
      print("clear weight of Mask Conv!")
   end
   self.weight:cmul(self.mask)
end

function MaskConv:reset()
end


function MaskConv:zeroGradParameters()
   self.conv:zeroGradParameters()
end

function MaskConv:updateParameters(learningRate)
   self.conv:updateParameters(learningRate)
end
