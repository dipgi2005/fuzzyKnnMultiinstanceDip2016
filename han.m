function dista=han(train_bagsA,train_bagsB,y)
            if(y==1)
                dista=hansf(train_bagsA, train_bagsB);
            else if(y==2)
                    dista=mindist(train_bagsA, train_bagsB);
            else if(y==3)                    
                    dista=surjectdist(train_bagsA, train_bagsB);
            else if(y==4)                    
                    dista=hansfavg(train_bagsA, train_bagsB);
                end
                end
                end
            end
end