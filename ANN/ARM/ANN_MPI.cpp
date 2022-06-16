#include "ANN_MPI.h"

ANN_MPI::ANN_MPI ( int* _num_each_layer, int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
{
    num_layers = _num_layers;
    study_rate = _study_rate;
    num_each_layer = new int[_num_layers + 2];
    num_each_layer[0] = _num_each_layer[0];
    for (int i = 1; i <= num_layers + 1; i++)
    {
        num_each_layer[i] = _num_each_layer[i];
        layers.push_back (new Layer (num_each_layer[i - 1], (num_each_layer[i]) ) );
    }
    //display();
    num_epoch = _num_epoch;
    batch_size = _batch_size;
}

ANN_MPI::~ANN_MPI()
{
    //printf ("begin free ANN_MPI");
    delete[]num_each_layer;
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    // printf ("free ANN_MPI\n");
}
void ANN_MPI::shuffle (const int num_sample, float** _trainMat, float** _labelMat)
{
    //init
    int *shuffle_index = new int[num_sample];
    float **trainMat_old = new float*[num_sample];
    float **labelMat_old = new float*[num_sample];
    for (int i = 0; i < num_sample; i++)
    {
        trainMat_old[i] = new float[num_each_layer[0]];
        labelMat_old[i] = new float[num_each_layer[num_layers + 1]];
    }
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_each_layer[0]; j++)
            trainMat_old[i][j] = _trainMat[i][j];
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            labelMat_old[i][j] = _labelMat[i][j];
    }

    //shuffle
    for (int i = 0; i < num_sample; i++) shuffle_index[i] = i;
    random_shuffle (shuffle_index, shuffle_index + num_sample);
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_each_layer[0]; j++) _trainMat[i][j] = trainMat_old[shuffle_index[i]][j];
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++) _labelMat[i][j] = labelMat_old[shuffle_index[i]][j];
    }

    //delete
    for (int i = 0; i < num_sample; i++)
    {
        delete []trainMat_old[i];
        delete []labelMat_old[i];
    }
    delete []trainMat_old;
    delete []labelMat_old;
    delete[] shuffle_index;

    printf ("finish shuffle\n");
}


void ANN_MPI::train (const int num_sample, float** _trainMat, float** _labelMat)
{
    printf ("begin training\n");
    float thre = 1e-2;
    float *avr_X = new float[num_each_layer[0]];
    float *avr_Y = new float[num_each_layer[num_layers + 1]];
    
    struct timespec sts,ets;
	long long dsec=0;
	long long dnsec=0;

    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        if (epoch % 50 == 0)
        {
            //  printf ("round%d:\n", epoch);
        }
        int index = 0;

        while (index < num_sample)
        {
            for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
            for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] = 0.0;

            for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            {
                layers[num_layers]->delta[j] = 0.0;
            }

            for (int batch_i = 0; batch_i < batch_size && index < num_sample; batch_i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
            {
                //前向传播
                for (int i = 0; i < num_each_layer[1]; i++)
                {
                    layers[0]->output_nodes[i] = 0.0;
                    for (int j = 0; j < num_each_layer[0]; j++)
                    {
                        layers[0]->output_nodes[i] += layers[0]->weights[i][j] * _trainMat[index][j];
                    }
                    layers[0]->output_nodes[i] += layers[0]->bias[i];
                    layers[0]->output_nodes[i] = layers[0]->activation_function (layers[0]->output_nodes[i]);
                }

                //QueryPerformanceCounter ( (LARGE_INTEGER*) &head); // start time
                timespec_get(&sts, TIME_UTC);
                for (int i_layer = 1; i_layer <= num_layers; i_layer++)
                {
                    for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
                    {
                        layers[i_layer]->output_nodes[i] = 0.0;
                        for (int j = 0; j < num_each_layer[i_layer]; j++)
                        {
                            layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
                        }
                        layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
                        layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function (layers[i_layer]->output_nodes[i]);

                    }

                }
                //QueryPerformanceCounter ( (LARGE_INTEGER*) &tail); // end time
                timespec_get(&ets, TIME_UTC);
                dsec+=ets.tv_sec-sts.tv_sec;
	            dnsec+=ets.tv_nsec-sts.tv_nsec;

//                static int tt = 0;
//                float loss = 0.0;
//                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
//                {
//                    loss += (layers[num_layers]->output_nodes[t] - _labelMat[index][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[index][t]);
//                }
//                printf ("第%d次训练：%0.12f\n", tt++,loss);


                //for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
                //printf ("%f,", layers[num_layers]->output_nodes[i]);
                //printf ("\n");
                // printf ("finish predict\n");

                //计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数,batch内取平均
                    layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
                }
                // printf ("finish cal error\n");
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _trainMat[index][X_i];
                for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] += _labelMat[index][Y_i];
            }

            //delta在batch内取平均，avr_X、avr_Y分别为本个batch的输入、输出向量的平均值
            if (index % batch_size == 0)
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= batch_size;
                    //for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
                    avr_Y[j] /= batch_size;
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

            }
            else
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (index % batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= (index % batch_size);
                    avr_Y[j] /= (index % batch_size);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
            }
            // printf("index:%d\n",index);
            //计算loss func，仅用于输出
            if ( index >= num_sample)
            {
                static int tt = 0;
                float loss = 0.0;
                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
                {
                    loss += (layers[num_layers]->output_nodes[t] - avr_Y[t]) * (layers[num_layers]->output_nodes[t] - avr_Y[t]);
                }
                //  printf ("第%d次训练：%0.12f\n", tt++, loss);
            }

            //反向传播更新参数

            //计算每层的delta,行访问优化
            for (int i = num_layers - 1; i >= 0; i--)
            {
                float *error = new float[num_each_layer[i + 1]];
                for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
                for (int k = 0; k < num_each_layer[i + 2]; k++)
                {
                    for (int j = 0; j < num_each_layer[i + 1]; j++)
                    {
                        error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
                    }
                }
                for (int j = 0; j < num_each_layer[i + 1]; j++)
                {
                    layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function (layers[i]->output_nodes[j]);
                }
                delete[]error;
            }

            //反向传播，weights和bias更新
            for (int k = 0; k < num_each_layer[1]; k++)
            {
                for (int j = 0; j < num_each_layer[0]; j++)
                {
                    layers[0]->weights[k][j] -= study_rate * avr_X[j] * layers[0]->delta[k];
                }
                layers[0]->bias[k] -= study_rate * layers[0]->delta[k];
            }
            for (int i = 1; i <= num_layers; i++)
            {
                for (int k = 0; k < num_each_layer[i + 1]; k++)
                {
                    for (int j = 0; j < num_each_layer[i]; j++)
                    {
                        layers[i]->weights[k][j] -= study_rate * layers[i - 1]->output_nodes[j] * layers[i]->delta[k];
                    }
                    layers[i]->bias[k] -= study_rate * layers[i]->delta[k];
                }
            }
            //printf ("finish bp with index:%d\n",index);
        }

        // display();
    }
    printf ("finish training\n");

	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("ori:%lld.%09llds\n",dsec,dnsec);
    //cout << "ori:" << time_ori * 1.0  / freq << "s" << endl;
    delete[]avr_X;
    delete[]avr_Y;
}

void ANN_MPI::train_MPI_all_static(const int num_sample, float** _trainMat, float** _labelMat)
{
	//对此函数中全部关键循环均进行了MPI优化，采用静态分配方式
	printf("begin training\n");   // cout<<' ';
	float thre = 1e-2;
	float* avr_X = new float[num_each_layer[0]];
	float* avr_Y = new float[num_each_layer[num_layers + 1]];


    struct timespec sts,ets;
	long long dsec=0;
	long long dnsec=0;
    timespec_get(&sts, TIME_UTC);

	//QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		if (epoch % 50 == 0)
		{
			//  printf ("round%d:\n", epoch);
		}
		int index = 0;

		while (index < num_sample)
		{
			for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
			for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] = 0.0;

			for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
			{
				layers[num_layers]->delta[j] = 0.0;
			}



			for (int batch_i = 0; batch_i < batch_size && index < num_sample; batch_i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
			{
				//1.前向传播，对i进行分配

				int i_size = (num_each_layer[1] + numprocs - 2) / (numprocs - 1);
				if (myid == 0)
				{
					for (int procs_i = 1; procs_i < numprocs; procs_i++)
					{
						for (int i = (procs_i - 1) * i_size; i < min(num_each_layer[0 + 1], procs_i * i_size); i++)
						{
							MPI_Send(layers[0]->weights[i], num_each_layer[0], MPI_FLOAT, procs_i, 99 + i, MPI_COMM_WORLD);
						}
					}
					for (int procs_i = 1; procs_i < numprocs; procs_i++)
					{
						MPI_Recv(layers[0]->output_nodes + (procs_i - 1) * i_size, i_size, MPI_FLOAT, procs_i, 97, MPI_COMM_WORLD, &status);
					}
				}
				else
				{
					for (int i = (myid - 1) * i_size; i < min(num_each_layer[0 + 1], myid * i_size); i++)
					{
						MPI_Recv(layers[0]->weights[i], num_each_layer[0], MPI_FLOAT, 0, 99 + i, MPI_COMM_WORLD, &status);
						layers[0]->output_nodes[i] = 0.0;
						for (int j = 0; j < num_each_layer[0]; j++)
						{
							layers[0]->output_nodes[i] += layers[0]->weights[i][j] * _trainMat[index][j];
						}
						layers[0]->output_nodes[i] += layers[0]->bias[i];
						layers[0]->output_nodes[i] = layers[0]->activation_function(layers[0]->output_nodes[i]);
					}
					MPI_Send(layers[0]->output_nodes + (myid - 1) * i_size, i_size, MPI_FLOAT, 0, 97, MPI_COMM_WORLD);
				}

				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
				{
					int i_size = (num_each_layer[i_layer + 1] + numprocs - 2) / (numprocs - 1);
					if (myid == 0)
					{
						for (int procs_i = 1; procs_i < numprocs; procs_i++)
						{
							MPI_Send(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, procs_i, 98, MPI_COMM_WORLD);
							for (int i = (procs_i - 1) * i_size; i < min(num_each_layer[i_layer + 1], procs_i * i_size); i++)
							{
								MPI_Send(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, procs_i, 1000 + i, MPI_COMM_WORLD);
							}
						}
						for (int procs_i = 1; procs_i < numprocs; procs_i++)
						{
							MPI_Recv(layers[i_layer]->output_nodes + (procs_i - 1) * i_size, i_size, MPI_FLOAT, procs_i, 97, MPI_COMM_WORLD, &status);
						}
					}
					else
					{
						MPI_Recv(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, 98, MPI_COMM_WORLD, &status);
						for (int i = (myid - 1) * i_size; i < min(num_each_layer[i_layer + 1], myid * i_size); i++)
						{
							MPI_Recv(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, 1000 + i, MPI_COMM_WORLD, &status);
							layers[i_layer]->output_nodes[i] = 0.0;
							for (int j = 0; j < num_each_layer[i_layer]; j++)
							{
								layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
							}
							layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
							layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(layers[i_layer]->output_nodes[i]);
						}
						MPI_Send(layers[i_layer]->output_nodes + (myid - 1) * i_size, i_size, MPI_FLOAT, 0, 97, MPI_COMM_WORLD);
					}

				}

				if (myid != 0)
					continue;

				// cout<<"finish pridect"<<endl;

				//计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数,batch内取平均
					layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]);
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
				}
				// printf ("finish cal error\n");
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _trainMat[index][X_i];
				for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] += _labelMat[index][Y_i];
			}

			//delta在batch内取平均，avr_X、avr_Y分别为本个batch的输入、输出向量的平均值
			if (index % batch_size == 0)
			{
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					if (batch_size == 0) printf("wrong!\n");
					layers[num_layers]->delta[j] /= batch_size;
					//for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
					avr_Y[j] /= batch_size;
				}
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

			}
			else
			{
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					if (index % batch_size == 0) printf("wrong!\n");
					layers[num_layers]->delta[j] /= (index % batch_size);
					avr_Y[j] /= (index % batch_size);
				}
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
			}
			// printf("index:%d\n",index);
			//计算loss func，仅用于输出
			if (index >= num_sample)
			{
				static int tt = 0;
				float loss = 0.0;
				for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
				{
					loss += (layers[num_layers]->output_nodes[t] - avr_Y[t]) * (layers[num_layers]->output_nodes[t] - avr_Y[t]);
				}
				// printf ("第%d次训练：%0.12f\n", tt++, loss);
			}

			//反向传播更新参数

			//2.计算每层的delta,行访问优化，对j进行数据分配
			for (int i = num_layers - 1; i >= 0; i--)
			{
				//从0号进程广播
				for (int k = 0; k < num_each_layer[i + 2]; k++)
				{
					MPI_Bcast(layers[i + 1]->weights[k], num_each_layer[i + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
				}
				MPI_Bcast(layers[i + 1]->delta, num_each_layer[i + 2], MPI_FLOAT, 0, MPI_COMM_WORLD);

				float* error = new float[num_each_layer[i + 1]];
				//0号进程也参与计算
				int i_size = (num_each_layer[i + 1] + numprocs - 1) / (numprocs);
				for (int j = myid * i_size; j < min(num_each_layer[i + 1], (myid + 1) * i_size); j++)
				{
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						error[j] = 0.0;
						error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
						layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
					}
				}
				//将数据汇总到0号进程
				if (myid == 0)
				{
					for (int procs_i = 1; procs_i < numprocs; procs_i++)
					{
						MPI_Recv(layers[i]->delta + procs_i * i_size, i_size, MPI_FLOAT, procs_i, 96, MPI_COMM_WORLD, &status);
					}
				}
				else
				{
					MPI_Send(layers[i]->delta + myid * i_size, i_size, MPI_FLOAT, 0, 96, MPI_COMM_WORLD);
				}

				delete[]error;
			}
			//cout<<"finish cal delta"<<endl;

						//3.反向传播，weights和bias更新，对k进行数据分配

			MPI_Bcast(layers[0]->delta, num_each_layer[1], MPI_FLOAT, 0, MPI_COMM_WORLD);//从0号进程广播
			int i_size = (num_each_layer[1] + numprocs - 1) / (numprocs);//0号进程也参与计算
			for (int k = myid * i_size; k < min(num_each_layer[1], (myid + 1) * i_size); k++)
			{
				for (int j = 0; j < num_each_layer[0]; j++)
				{
					layers[0]->weights[k][j] -= study_rate * avr_X[j] * layers[0]->delta[k];
				}
				layers[0]->bias[k] -= study_rate * layers[0]->delta[k];
			}
			//将数据汇总到0号进程
			if (myid == 0)
			{
				for (int procs_i = 1; procs_i < numprocs; procs_i++)
				{
					MPI_Recv(layers[0]->bias + procs_i * i_size, i_size, MPI_FLOAT, procs_i, 94, MPI_COMM_WORLD, &status);
					for (int k = procs_i * i_size; k < min(num_each_layer[1], (procs_i + 1) * i_size); k++)
					{
						MPI_Recv(layers[0]->weights[k], num_each_layer[0], MPI_FLOAT, procs_i, 3000 + k, MPI_COMM_WORLD, &status);
					}
				}
			}
			else
			{
				MPI_Send(layers[0]->bias + myid * i_size, i_size, MPI_FLOAT, 0, 94, MPI_COMM_WORLD);
				for (int k = myid * i_size; k < min(num_each_layer[1], (myid + 1) * i_size); k++)
				{
					MPI_Send(layers[0]->weights[k], num_each_layer[0], MPI_FLOAT, 0, 3000 + k, MPI_COMM_WORLD);
				}
			}
			//cout<<"finish first bp"<<endl;
						//同理
			for (int i = 1; i <= num_layers; i++)
			{
				MPI_Bcast(layers[i]->delta, num_each_layer[i + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
				MPI_Bcast(layers[i - 1]->output_nodes, num_each_layer[i], MPI_FLOAT, 0, MPI_COMM_WORLD);
				int i_size = (num_each_layer[i + 1] + numprocs - 1) / (numprocs);
				// cout << "i_size" << i_size<<' '<< num_each_layer[i + 1]<<' '<< numprocs<<endl;

				for (int k = myid * i_size; k < min(num_each_layer[i + 1], (myid + 1) * i_size); k++)
				{
					for (int j = 0; j < num_each_layer[i]; j++)
					{
						layers[i]->weights[k][j] -= study_rate * layers[i - 1]->output_nodes[j] * layers[i]->delta[k];
					}
					layers[i]->bias[k] -= study_rate * layers[i]->delta[k];
				}
				if (myid == 0)
				{
					for (int procs_i = 1; procs_i < numprocs; procs_i++)
					{
						MPI_Recv(layers[i]->bias + procs_i * i_size, i_size, MPI_FLOAT, procs_i, 95, MPI_COMM_WORLD, &status);
						for (int k = procs_i * i_size; k < min(num_each_layer[i + 1], (procs_i + 1) * i_size); k++)
						{
							MPI_Recv(layers[i]->weights[k], num_each_layer[i], MPI_FLOAT, procs_i, 2000 + k, MPI_COMM_WORLD, &status);
						}
					}
				}
				else
				{
					MPI_Send(layers[i]->bias + myid * i_size, i_size, MPI_FLOAT, 0, 95, MPI_COMM_WORLD);
					for (int k = myid * i_size; k < min(num_each_layer[i + 1], (myid + 1) * i_size); k++)
					{
						MPI_Send(layers[i]->weights[k], num_each_layer[i], MPI_FLOAT, 0, 2000 + k, MPI_COMM_WORLD);
					}
				}

			}
			//printf ("finish bp with index:%d\n",index);
		}
		// display();
	}
    timespec_get(&ets, TIME_UTC);
    dsec+=ets.tv_sec-sts.tv_sec;
	dnsec+=ets.tv_nsec-sts.tv_nsec;
	printf("finish training\n");
	//std::cout << myid << "mpi_all:" << time_mpi * 1.0 / freq << "s" << endl;
    printf ("mpi_all:%lld.%09llds\n",dsec,dnsec);
	delete[]avr_X;
	delete[]avr_Y;
}

void ANN_MPI::predict_MPI_static1() {
	//静态的朴素优化方法
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
		int i_size = (num_each_layer[i_layer + 1] + numprocs - 2) / (numprocs - 1);//0号进程不参与计算
	   //数据发送
		if (myid == 0)
		{
			for (int procs_i = 1; procs_i < numprocs; procs_i++)
			{
				MPI_Send(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, procs_i, 98, MPI_COMM_WORLD);
				for (int i = (procs_i - 1) * i_size; i < min(num_each_layer[i_layer + 1], procs_i * i_size); i++)
				{
					MPI_Send(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, procs_i, 99 + i, MPI_COMM_WORLD);
				}

				//  cout << "0 end send" << endl;
			}
			for (int procs_i = 1; procs_i < numprocs; procs_i++)
			{
				MPI_Recv(layers[i_layer]->output_nodes + (procs_i - 1) * i_size, i_size, MPI_FLOAT, procs_i, 97, MPI_COMM_WORLD, &status);
			}
			// cout<<myid<<"finish recv"<<endl;

		}
		else
		{
			MPI_Recv(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, 98, MPI_COMM_WORLD, &status);
			//从节点计算
			for (int i = (myid - 1) * i_size; i < min(num_each_layer[i_layer + 1], myid * i_size); i++)
			{
				MPI_Recv(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, 99 + i, MPI_COMM_WORLD, &status);
				layers[i_layer]->output_nodes[i] = 0.0;
				for (int j = 0; j < num_each_layer[i_layer]; j++)
				{
					layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
				}
				layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
				layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(layers[i_layer]->output_nodes[i]);
			}
			//从节点将数据发送回主节点（0号）
			MPI_Send(layers[i_layer]->output_nodes + (myid - 1) * i_size, i_size, MPI_FLOAT, 0, 97, MPI_COMM_WORLD);
			// cout<<myid<<"finish send"<<endl;
		}

	}
}
void ANN_MPI::predict_MPI_gather() {
	//多进程规约
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
	   //数据发送
		int position = 0;
		int buff_size = num_each_layer[i_layer + 1] * num_each_layer[i_layer] * sizeof(float);
		float* buffer_for_packed=new float[buff_size];
		float* recv_buffer=new float[buff_size];

		if(myid==0){
            for (int i = 0; i < num_each_layer[i_layer+1]; i++) {
                MPI_Pack(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, buffer_for_packed, buff_size, &position, MPI_COMM_WORLD);
            }
		}

		int i_size = (num_each_layer[i_layer + 1] + numprocs - 1) / (numprocs);//0号进程参与计算
		MPI_Scatter(buffer_for_packed, num_each_layer[i_layer] * i_size, MPI_PACKED, recv_buffer, num_each_layer[i_layer]*i_size, MPI_PACKED,0, MPI_COMM_WORLD);
		MPI_Bcast(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);

		if (myid != 0) {
			position = 0;
			for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++) {
				MPI_Unpack( recv_buffer, buff_size,&position,layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT,MPI_COMM_WORLD);
			}
		}
        delete[]buffer_for_packed;
		delete[]recv_buffer;
		//计算
		for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++)
		{
			layers[i_layer]->output_nodes[i] = 0.0;
			for (int j = 0; j < num_each_layer[i_layer]; j++)
			{
				layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
			}
			layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
			layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(layers[i_layer]->output_nodes[i]);
		}

		//数据收集
		float* buff_output = new float[num_each_layer[i_layer + 1]];
		MPI_Gather(layers[i_layer]->output_nodes + myid * i_size, i_size, MPI_FLOAT, buff_output, i_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if(myid==0)
            memcpy(layers[i_layer]->output_nodes,buff_output,num_each_layer[i_layer + 1]);
		delete[] buff_output;
	}
}

void ANN_MPI::predict_MPI_alltoall() {
	//多进程规约
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
	   //数据发送
		int position = 0;
		int buff_size = num_each_layer[i_layer + 1] * num_each_layer[i_layer] * sizeof(float);
		float* buffer_for_packed=new float[buff_size];
		float* recv_buffer=new float[buff_size];

		if(myid==0){
            for (int i = 0; i < num_each_layer[i_layer+1]; i++) {
                MPI_Pack(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, buffer_for_packed, buff_size, &position, MPI_COMM_WORLD);
            }
		}

		int i_size = (num_each_layer[i_layer + 1] + numprocs - 1) / (numprocs);//0号进程参与计算
		MPI_Scatter(buffer_for_packed, num_each_layer[i_layer] * i_size, MPI_PACKED, recv_buffer, num_each_layer[i_layer]*i_size, MPI_PACKED,0, MPI_COMM_WORLD);
		MPI_Bcast(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);

		if (myid != 0) {
			position = 0;
			for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++) {
				MPI_Unpack( recv_buffer, buff_size,&position,layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT,MPI_COMM_WORLD);
			}
		}
        delete[]buffer_for_packed;
		delete[]recv_buffer;
		//计算
		for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++)
		{
			layers[i_layer]->output_nodes[i] = 0.0;
			for (int j = 0; j < num_each_layer[i_layer]; j++)
			{
				layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
			}
			layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
			layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(layers[i_layer]->output_nodes[i]);
		}

		//数据收集
		float* buff_output = new float[num_each_layer[i_layer + 1]];
		MPI_Gather(layers[i_layer]->output_nodes + myid * i_size, i_size, MPI_FLOAT, buff_output, i_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if(myid==0)
            memcpy(layers[i_layer]->output_nodes,buff_output,num_each_layer[i_layer + 1]);
		delete[] buff_output;
	}
}

void ANN_MPI::predict_MPI_static2() {
	//静态的优化方法,接受全部输入
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
		int i_size = (num_each_layer[i_layer + 1] + numprocs - 1) / (numprocs);//0号进程参与计算
	   //数据发送
		MPI_Bcast(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
		for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
			MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
		//计算
		for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++)
		{
			layers[i_layer]->output_nodes[i] = 0.0;
			for (int j = 0; j < num_each_layer[i_layer]; j++)
			{
				layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
			}
			layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
			layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(layers[i_layer]->output_nodes[i]);
		}
		if (myid == 0)
		{
			float* temp_nodes = new float[num_each_layer[i_layer + 1]];
			for (int temp_procs_i = 1; temp_procs_i < numprocs; temp_procs_i++)
			{
				MPI_Recv(temp_nodes, i_size, MPI_FLOAT, MPI_ANY_SOURCE, 97, MPI_COMM_WORLD, &status);
				memcpy(layers[i_layer]->output_nodes + status.MPI_SOURCE * i_size, temp_nodes, sizeof(float) * i_size);
			}
			delete[] temp_nodes;
			// cout<<myid<<"finish recv"<<endl;
		}
		else
		{
			//从节点将数据发送回主节点（0号）
			MPI_Send(layers[i_layer]->output_nodes + myid * i_size, i_size, MPI_FLOAT, 0, 97, MPI_COMM_WORLD);
			// cout<<myid<<"finish send"<<endl;
		}
	}
}
void ANN_MPI::predict_MPI_dynamic() {
	//动态分配任务,主从式,0号进程不参与计算
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
		//数据发送
		//int g = 1;粒度必须为1行，因为多行的接收不能保证为原子操作
		//广播layers[i_layer - 1]->output_nodes,
		MPI_Bcast(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (myid == 0)
		{
			int i;
			for (i = 0; i < numprocs - 1; i++) {
				//for (int g_i = 0; g_i < g; g_i++)
				MPI_Send(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, i + 1, i, MPI_COMM_WORLD);//发送一行，tag为行号
			}
			float temp_node;
			int finish = 0;
			while (finish < numprocs - 1) {
				//cout << "finish"<<finish << endl;
				//第i-1行已完成
				MPI_Recv(&temp_node, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				//memcpy(layers[i_layer]->output_nodes + status.MPI_TAG * g, temp_nodes, sizeof(float) * g);
				if (status.MPI_TAG < num_each_layer[i_layer + 1])
					layers[i_layer]->output_nodes[status.MPI_TAG] = temp_node;

				if (i < num_each_layer[i_layer + 1]) {//根据tag判断是否完成全部的行
					//发送新的一行，i
					MPI_Send(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, status.MPI_SOURCE, i, MPI_COMM_WORLD);
					i++;
				}
				else {
					//发送任意数据，使从进程退出循环
					//cout << "finish" << finish << endl;
					MPI_Send(layers[i_layer]->weights[0], num_each_layer[i_layer], MPI_FLOAT, status.MPI_SOURCE, num_each_layer[i_layer + 1] + 1, MPI_COMM_WORLD);
					finish++;
				}
			}
		}
		else
		{
			float* temp_nodes = new float[num_each_layer[i_layer]];
			int i = myid;
			while (i < num_each_layer[i_layer + 1]) {
				MPI_Recv(temp_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				i = status.MPI_TAG;
				if (i >= num_each_layer[i_layer + 1]) break;
				//计算
				float output_nodes = 0.0;
				for (int j = 0; j < num_each_layer[i_layer]; j++)
				{
					output_nodes += temp_nodes[j] * layers[i_layer - 1]->output_nodes[j];
				}
				output_nodes += layers[i_layer]->bias[i];
				output_nodes = layers[i_layer]->activation_function(output_nodes);

				//从节点将数据发送回主节点（0号）
				MPI_Send(&output_nodes, 1, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
			}
			delete[] temp_nodes;
		}
		//cout<<myid<<"finish pridect"<< i_layer<<endl;
	}
}


void ANN_MPI::predict_MPI_threads() {
	//静态的优化方法,接受全部输入,MPI+openMP
	int myid, numprocs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int i_layer = 1; i_layer <= num_layers; i_layer++)
	{
		int i_size = (num_each_layer[i_layer + 1] + numprocs - 1) / (numprocs);//0号进程参与计算
	   //数据发送
		MPI_Bcast(layers[i_layer - 1]->output_nodes, num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
		for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
			MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
		//计算
#pragma omp parallel num_threads(NUM_THREADS)
		{
			for (int i = myid * i_size; i < min(num_each_layer[i_layer + 1], (myid + 1) * i_size); i++)
			{
				float sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
				for (int j = 0; j < num_each_layer[i_layer]; j++)
				{
					sum += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
				}
				sum += layers[i_layer]->bias[i];
				layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(sum);
			}
		}
		if (myid == 0)
		{
			float* temp_nodes = new float[num_each_layer[i_layer + 1]];
			for (int temp_procs_i = 1; temp_procs_i < numprocs; temp_procs_i++)
			{
				MPI_Recv(temp_nodes, i_size, MPI_FLOAT, MPI_ANY_SOURCE, 97, MPI_COMM_WORLD, &status);
				memcpy(layers[i_layer]->output_nodes + status.MPI_SOURCE * i_size, temp_nodes, sizeof(float) * i_size);
			}
			delete[] temp_nodes;
			// cout<<myid<<"finish recv"<<endl;
		}
		else
		{
			//从节点将数据发送回主节点（0号）
			MPI_Send(layers[i_layer]->output_nodes + myid * i_size, i_size, MPI_FLOAT, 0, 97, MPI_COMM_WORLD);
			// cout<<myid<<"finish send"<<endl;
		}
	}
}

void ANN_MPI::train_MPI_predict(const int num_sample, float** _trainMat, float** _labelMat){
	//仅对正向传播循环进行了MPI优化
	//printf("begin training\n");

	int myid, numprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	float thre = 1e-2;
	float* avr_X = new float[num_each_layer[0]];
	float* avr_Y = new float[num_each_layer[num_layers + 1]];


	//long long time_mpi1 = 0, time_mpi2 = 0, time_mpi3 = 0, time_mpi4=0;
    struct timespec sts,ets;
	long long dsec1=0,dsec2=0,dsec3=0;
	long long dnsec1=0,dnsec2=0,dnsec3=0;
	//long long head, tail, freq;// timers
	//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		if (epoch % 50 == 0)
		{
			//  printf ("round%d:\n", epoch);
		}
		int index = 0;

		while (index < num_sample)
		{
			for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
			for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] = 0.0;

			for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
			{
				layers[num_layers]->delta[j] = 0.0;
			}

			for (int batch_i = 0; batch_i < batch_size && index < num_sample; batch_i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
			{
				//前向传播

				for (int i = 0; i < num_each_layer[1]; i++)
				{
					layers[0]->output_nodes[i] = 0.0;
					for (int j = 0; j < num_each_layer[0]; j++)
					{
						layers[0]->output_nodes[i] += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					layers[0]->output_nodes[i] += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(layers[0]->output_nodes[i]);
				}
				MPI_Barrier(MPI_COMM_WORLD);
                timespec_get(&sts, TIME_UTC);
	            predict_MPI_static1();
                timespec_get(&ets, TIME_UTC);
                dsec1+=ets.tv_sec-sts.tv_sec;
	            dnsec1+=ets.tv_nsec-sts.tv_nsec;

				MPI_Barrier(MPI_COMM_WORLD);
                timespec_get(&sts, TIME_UTC);
	            predict_MPI_static2();
                timespec_get(&ets, TIME_UTC);
                dsec2+=ets.tv_sec-sts.tv_sec;
	            dnsec2+=ets.tv_nsec-sts.tv_nsec;
                
                MPI_Barrier(MPI_COMM_WORLD);
                timespec_get(&sts, TIME_UTC);
	            predict_MPI_dynamic();
                timespec_get(&ets, TIME_UTC);
                dsec3+=ets.tv_sec-sts.tv_sec;
	            dnsec3+=ets.tv_nsec-sts.tv_nsec;
				MPI_Barrier(MPI_COMM_WORLD);

				if (myid != 0)
					continue;

				//计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数,batch内取平均
					layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]);
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
				}
				// printf ("finish cal error\n");
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _trainMat[index][X_i];
				for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] += _labelMat[index][Y_i];
			}

			//delta在batch内取平均，avr_X、avr_Y分别为本个batch的输入、输出向量的平均值
			if (index % batch_size == 0)
			{
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					if (batch_size == 0) printf("wrong!\n");
					layers[num_layers]->delta[j] /= batch_size;
					//for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
					avr_Y[j] /= batch_size;
				}
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

			}
			else
			{
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					if (index % batch_size == 0) printf("wrong!\n");
					layers[num_layers]->delta[j] /= (index % batch_size);
					avr_Y[j] /= (index % batch_size);
				}
				for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
			}
			// printf("index:%d\n",index);
			//计算loss func，仅用于输出
			if (index >= num_sample)
			{
				static int tt = 0;
				float loss = 0.0;
				for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
				{
					loss += (layers[num_layers]->output_nodes[t] - avr_Y[t]) * (layers[num_layers]->output_nodes[t] - avr_Y[t]);
				}
				// printf ("第%d次训练：%0.12f\n", tt++, loss);
			}

			//反向传播更新参数

			//计算每层的delta,行访问优化
			for (int i = num_layers - 1; i >= 0; i--)
			{
				float* error = new float[num_each_layer[i + 1]];
				for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
				for (int k = 0; k < num_each_layer[i + 2]; k++)
				{
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
					}
				}
				for (int j = 0; j < num_each_layer[i + 1]; j++)
				{
					layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
				}
				delete[]error;
			}

			//反向传播，weights和bias更新
			for (int k = 0; k < num_each_layer[1]; k++)
			{
				for (int j = 0; j < num_each_layer[0]; j++)
				{
					layers[0]->weights[k][j] -= study_rate * avr_X[j] * layers[0]->delta[k];
				}
				layers[0]->bias[k] -= study_rate * layers[0]->delta[k];
			}

			for (int i = 1; i <= num_layers; i++)
			{
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					for (int j = 0; j < num_each_layer[i]; j++)
					{
						layers[i]->weights[k][j] -= study_rate * layers[i - 1]->output_nodes[j] * layers[i]->delta[k];
					}
					layers[i]->bias[k] -= study_rate * layers[i]->delta[k];
				}
			}
			//printf ("finish bp with index:%d\n",index);
		}
		// display();
	}

	//MPI_Finalize();
	if(myid==0){
        if (dnsec1<0){dsec1--;dnsec1+=1000000000ll;}
        if (dnsec2<0){dsec2--;dnsec2+=1000000000ll;}
        if (dnsec3<0){dsec3--;dnsec3+=1000000000ll;}
        printf ("mpi_1:%lld.%09llds\n",dsec1,dnsec1);
     	printf ("mpi_2:%lld.%09llds\n",dsec2,dnsec2);
        printf ("mpi_3:%lld.%09llds\n",dsec3,dnsec3);
	}
	//printf("finish training\n");

	delete[]avr_X;
	delete[]avr_Y;
}



bool ANN_MPI::isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
{
    float lossFunc = 0.0;
    for (int k = 0; k < _sampleNum; ++k)
    {
        predict (_trainMat[k]);
        float loss = 0.0;
        for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
        {
            loss += (layers[num_layers]->output_nodes[t] - _labelMat[k][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[k][t]);
        }
        lossFunc += (1.0 / 2) * loss;
    }

    lossFunc = lossFunc / _sampleNum;

    static int tt = 0;
    tt++;
    // printf ("第%d次训练：%0.12f\n", tt, lossFunc);

    if (lossFunc > _thresh) return true;
    return false;
}
void ANN_MPI::predict (float* in)
{
    layers[0]->_forward (in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward (layers[i - 1]->output_nodes);
    }
}
void ANN_MPI::display()
{
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}
void ANN_MPI::get_predictions (float* X)
{
    predict (X);
    static int t = 0;
    printf ("in%d:", t);
    for (int i = 0; i < min (5, num_each_layer[0]); i++) printf ("%f,", X[i]);
    printf ("  out%d:", t);
    for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
        printf ("%f,", layers[num_layers]->output_nodes[i]);
    t++;
    printf ("\n");
}
