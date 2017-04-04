function [y,fs,bits]=wavread(filename)

    [y,fs]=audioread(filename);
    bits=16;
    
