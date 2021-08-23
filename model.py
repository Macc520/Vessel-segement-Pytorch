# ==================================================================

class attention_block(nn.Module):
    def __init__(self,ch_in, ch_out):
          super(attention_block,self).__init__()
          self.sigmoid = nn.Sigmoid()
          self.conv = nn.Conv2d(ch_in, ch_out, 1)
          self.BN = nn.BatchNorm2d(ch_out)
    def forward(self,x,f):
        x1 = x
        x = self.sigmoid(x)
        f = f.permute(0,1,3,2)
        y = torch.mul(f,x)
        # y = y/self.BN(x)*self.BN(f)
        y = self.conv(y)
        y = x1+y
        return y

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x



class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.attention1 = attention_block(128,128)
        self.attention2 = attention_block(128,128)
        self.attention3 = attention_block(64,64)
        # self.attention1 = Attention_block(128,128,64)
        # self.attention2 = Attention_block(64,64,32)
        # self.attention3 = attention_block(64,64)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling1 = nn.Upsample(scale_factor=2)
        self.upsampling2 = nn.Upsample(scale_factor=2)
        self.upsampling3 = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.UP6 = up_conv(ch_in=128, ch_out=128)
        self.UP7 = up_conv(ch_in=256, ch_out=256)
        # self.Conv6 = conv_block(ch_in=128, ch_out=64)
        # self.Conv7 = conv_block(ch_in=256, ch_out=512)
        # self.Up_conv8 = conv_block(ch_in=512, ch_out=1024)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.nl1 = nn.Conv2d(192,128,(7,1),stride=(1, 1),padding=(3,0))    # 7x1卷积
        self.nl2 = nn.Conv2d(128,64,(1,7),stride=(1, 1),padding=(0,3))     # 1x7卷积
        
        # self.nl3 = nn.Conv2d(192,64,(3,3),stride=(1, 1),padding=(1,1))  
        # self.nl3 = nn.Conv2d(192,64,(7,7),stride=(1, 1),padding=(3,3)) 
        # self.nl3 = nn.Conv2d(192,64,(9,9),stride=(1, 1),padding=(4,4)) 
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    def forward(self, x):
      
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        Upx2 = self.Up2(x2) #64*48*48
        sum_1 = torch.cat((Upx2, x1), dim=1) #128*48*48
        # sum_1 = self.Up_conv8(sum_1)
        
        #print(sum_1.shape)
        # print(Upx2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4_1 = d4

        # upd4 = self.Up3(d4_1) 
        #print(upd4.shape) #1*128*24*24
        # upd4 = self.Up_conv3(upd4) 

        upd4 = self.upsampling1(d4_1) 
        upd4 = self.upsampling2(upd4) 
        #print(upd4.shape)#1*256*48*48
        upd4 = self.Up_conv3(upd4)
        #print(upd4.shape)
        atten1 = self.attention1(sum_1,upd4)
        #print(atten1.shape)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3_1 = d3

        upd3 = self.upsampling3(d3_1) 
        #print(upd4.shape) #1*128*24*24
        #print(upd4.shape)#1*64*48*48
        #upd4 = self.Conv6(upd4)
        #print(upd3.shape)
        atten2 = self.attention1(atten1,upd3)
        #print(atten2.shape)
        



        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2_1 = d2
        atten2 = self.Up_conv2(atten2)
        atten3 = self.attention3(atten2,d2_1)
        #print(atten3.shape)

        
        d1 = self.Conv_1x1(atten1)
        d1 = F.softmax(d1,dim=1)  # mine

        d2 = self.Conv_1x1(atten2)
        d2 = F.softmax(d2,dim=1) 

        d3 = self.Conv_1x1(atten3)
        d3 = F.softmax(d3,dim=1) 

        d4 = torch.cat((atten1,atten2,atten3),dim=1)
        d4 = self.nl1(d4)
        d4 = self.relu1(d4)
        d4 = self.nl2(d4)
        d4 = self.relu1(d4)
        d4 = self.Conv_1x1(d4)
        d4 = F.softmax(d4,dim=1) 

        return  d1,d2,d3,d4
