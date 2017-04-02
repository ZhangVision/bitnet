
local nn = require 'nn'
require 'cunn'
require 'math'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization



local function createModel(opt)
   local function bstack(block, nblock,depth, n_filter_in ,n_filter_out, stride)
      local s = nn.Sequential()
      for i=1,nblock do
         s:add(block(depth,i == 1 and n_filter_in or n_filter_out ,n_filter_out,i == 1 and stride or 1))
         if depth>1 then
            s:add(nn.JoinTable(2))
         end
      end
      return s
   end

   local function block(depth, n_filter_in,n_filter_out,stride)
      if depth==1 then
         return nn.Sequential()
                        :add(Convolution(n_filter_in,n_filter_out,3,3,stride,stride,1,1))
                        :add(SBatchNorm(n_filter_out))
                        :add(ReLU(true))
      end

      local cat = nn.ConcatTable()
      local s={}
      
      for i=1,depth do
         _stride= i == 1 and stride or 1
         nInputPlane= i==1 and n_filter_in or n_filter_out/math.pow(2,i-1)
         nOutPlane=n_filter_out/math.pow(2,i)
         if i>1 then
            s[i]= nn.Sequential()
                        :add(s[i-1])
                        :add(Convolution(nInputPlane,nOutPlane,3,3,_stride,_stride,1,1))
                        :add(SBatchNorm(nOutPlane))
                        :add(ReLU(true))
            cat:add(nn.Sequential()
                     :add(s[i-1])
                     :add(Convolution(nInputPlane,nOutPlane,3,3,_stride,_stride,1,1))
                     :add(SBatchNorm(nOutPlane))
                     :add(ReLU(true))
                     )
         else
            s[i]= nn.Sequential()
                        :add(Convolution(nInputPlane,nOutPlane,3,3,_stride,_stride,1,1))
                        :add(SBatchNorm(nOutPlane))
                        :add(ReLU(true))
            cat:add(nn.Sequential()
                     :add(Convolution(nInputPlane,nOutPlane,3,3,_stride,_stride,1,1))
                     :add(SBatchNorm(nOutPlane))
                     :add(ReLU(true))
                     )
         end
      end

      cat:add(s[depth])
      return cat
   end

   

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      if opt.depth==26 then
         model:add(bstack(block,2,3, 64,128,1))
         model:add(bstack(block,1,3, 128,192,2))
         model:add(bstack(block,1,3, 192,256,1))
         model:add(bstack(block,1,3, 256,384,2))
         model:add(bstack(block,1,3, 384,384,1))
         model:add(bstack(block,1,3, 384,512,2))
         model:add(bstack(block,1,3, 512,768,1))
      else
         model:add(bstack(block,2,4, 64,256,1))
         model:add(bstack(block,2,4, 256,384,2))
         model:add(bstack(block,2,4, 384,512,2))
         model:add(bstack(block,2,4, 512, 768, 2))
      end
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(768):setNumInputDims(3))
      model:add(nn.Linear(768, 1000))
   elseif opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      w=opt.width
      d=opt.bdepth
      n=opt.nblock
      model:add(bstack(block,n,d, 16,16*w,1))
      model:add(bstack(block,n,d, 16*w,32*w,2))
      model:add(bstack(block,n,d, 32*w, 64*w, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64*w):setNumInputDims(3))
      model:add(nn.Linear(64*w, opt.dataset == 'cifar10' and 10 or 100))  
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel

