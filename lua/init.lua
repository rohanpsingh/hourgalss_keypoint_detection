print("Initalizing lua scripts...")
--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------
require 'torch'
torch.setnumthreads(1)

require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

inImage_c1 = torch.FloatTensor()
inImage_c2 = torch.FloatTensor()
inImage_c3 = torch.FloatTensor()
inImage = torch.FloatTensor()

predHMs = torch.Tensor(1,6,64,64)

local min_hm_thresh = 0.5


function loadImage()

    inImage = torch.cat(inImage_c3, inImage_c2, 3):cat(inImage_c1, 3)
    local rgb = inImage:permute(3,1,2)
    local img = image.minmax{tensor = rgb}
    local out = image.toDisplayTensor{img}
    --print(torch.typename(out))
    --image.save("test.jpg", out)
    --print("saved image")
end


function loadModel(model)

    if model == 'pretrained' then
        m = torch.load('pose-hg-pascal3d.t7')   -- Load pre-trained model
    else
        m = torch.load(model)   -- Load the specified model
        m:evaluate()
    end
    print("loading model: ", model)
    m:cuda()
end


function evaluate(img_cx, img_cy, img_scale)

    -- Set up input image
    local im = inImage:permute(3,1,2)
    im = im/255
    local center = torch.DoubleTensor{img_cx, img_cy}
    local scale = img_scale
    local inp = crop(im, center, scale, 0, 256)

    -- Get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    out = applyFn(function (x) return x:clone() end, out)
    local flippedOut = m:forward(flip(inp:view(1,3,256,256):cuda()))
    flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
    out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)
    cutorch.synchronize()

    predHMs:copy(out[#out])

    keypoint_locs = torch.Tensor(predHMs:size(2), 2)

    for i=1,predHMs:size(2) do
        local kp = predHMs:select(2,i):select(1,1)
        local maxP_per_row, bestColumn_per_row = torch.max(kp,2)
        local best_p, best_row = torch.max(maxP_per_row, 1)
        local best_col = bestColumn_per_row[best_row[1][1]]
        local kpy = best_row[1][1]/64*200*scale + center[2] - scale*100
        local kpx = best_col[1]/64*200*scale + center[1] - scale*100
        if best_p[1][1] > min_hm_thresh then
            keypoint_locs[i][1] = kpx
            keypoint_locs[i][2] = kpy
        else
            keypoint_locs[i][1] = -1
            keypoint_locs[i][2] = -1
        end
    end

    --print(keypoint_locs)

    collectgarbage()

end
