--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 11/8/16
-- Time: 02:01
--

local MaskConv, Parent = torch.class('nn.MaskConv', 'nn.Module')

function MaskConv:__init(data_type, mask_type,
                         nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   -- data_type: `cudnn` or `nn`
   -- mask_type: `A` or 'B' or nil (none)
   --
   -- this module is masked before weight updates,
   -- technically, we apply a mask to `gradOutput` before `accGradParameters()`

   Parent.__init(self)

   if mask_type == 'A' then
      self.m = torch.Tensor(kH, kW):fill(1)
      self.m[{{(kH + 1)/2, kH}, {(kW + 1)/2, kW}}] = 0
      self.m[{{(kH + 1)/2 + 1, kH}, {1, (kW + 1)/2 - 1}}] = 0
   elseif mask_type == 'B' then
      assert(nil, "Please implement type B")
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

   self.weight, self.gradWeight = self.conv:parameter()

end

function MaskConv:updateOutput(input)
   self.conv:forward(input)
end

function MaskConv:updateGradInput(input, gradOutput)
   self.conv:updateGradInput(input, gradOutput)
end

function MaskConv:accGradParameters(input, gradOutput)
   gradOutput:cmul(self.mask)
   self.conv:accGradParameters(input, gradOutput)
end

function MaskConv:initWeight()
   print("init weight of Mask Conv!")
   self.weight:cmul(self.mask)
end

function MaskConv:reset()
end

