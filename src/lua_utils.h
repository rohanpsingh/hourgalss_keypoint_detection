#ifndef LUA_UTILS_H
#define LUA_UTILS_H

#include "common_headers.h"

//lua state
lua_State *L;


void loadImage(const cv::Mat input_image){

    cv::Mat image;
    input_image.convertTo(image, CV_32FC3);
    cv::Mat img_bgr[3];
    cv::split(image, img_bgr);

    const int height = input_image.rows;
    const int width = input_image.cols;
    const int channs = input_image.channels();
    const int size = channs*height*width;
    lua_getglobal(L, "inImage_c1");
    float* imgData = img_bgr[0].ptr<float>();
    THFloatStorage* imgStorage = THFloatStorage_newWithData(imgData, size);
    THFloatTensor* imgTensor = THFloatTensor_newWithStorage2d(imgStorage, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c1");
    lua_getglobal(L, "inImage_c2");
    float* imgData1 = img_bgr[1].ptr<float>();
    THFloatStorage* imgStorage1 = THFloatStorage_newWithData(imgData1, size);
    THFloatTensor* imgTensor1 = THFloatTensor_newWithStorage2d(imgStorage1, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor1, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c2");
    lua_getglobal(L, "inImage_c3");
    float* imgData2 = img_bgr[2].ptr<float>();
    THFloatStorage* imgStorage2 = THFloatStorage_newWithData(imgData2, size);
    THFloatTensor* imgTensor2 = THFloatTensor_newWithStorage2d(imgStorage2, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor2, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c3");

    lua_getglobal(L, "loadImage");
    lua_pcall(L,0,0,0);
    return;
}

void loadBBox(const int xmin, const int xmax, const int ymin, const int ymax){

    int cx = abs(xmax+xmin)/2;
    int cy = abs(ymax+ymin)/2;
    float scale = std::max(xmax-xmin, ymax-ymin);
    scale /= 200.0f;

    lua_getglobal(L, "evaluate");
    lua_pushinteger(L,cx);
    lua_pushinteger(L,cy);
    lua_pushnumber(L,scale);
    lua_pcall(L,3,0,1);
    return;
}

void getUdata(const char* var_name, THFloatTensor** varTensor, float** data){

    lua_getglobal(L, var_name);
    *varTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");
    *data = THFloatTensor_data(*varTensor);
    return;
}

#endif // LUA_UTILS_H
