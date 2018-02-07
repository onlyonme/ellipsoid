#define REAL float
#define INT int
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
using namespace std;

const REAL pi = 3.141592653589793238462643383;

INT threads = 8;
INT bernFlag = 0;
INT loadBinaryFlag = 0;
INT outBinaryFlag = 0;
INT trainTimes = 1000;
INT nbatches = 100;
INT dimension = 100;
REAL alpha = 0.00001;

string inPath = "./";
string outPath = "./";
string loadPath = "./";
string note = "";
string note1 = "";

struct Triple {
	INT h, r, t;
};


Triple *trainList;

unsigned long long *next_random;

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

REAL rand(REAL min, REAL max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

REAL normal(REAL x, REAL miu,REAL sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

REAL randn(REAL miu,REAL sigma, REAL min ,REAL max) {
	REAL x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

INT relationTotal, entityTotal, tripleTotal;

REAL *relationVec, *entityVec, *A,*matrix_h, *matrix_t, *centr_h, *centr_t,*L_h,*L_t,*etmp;

void init() {
	cout<<"inin start\n";
	FILE *fin;
	INT tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);
    int m = dimension * (dimension + 1) / 2;

	entityVec= (REAL *)calloc(entityTotal * dimension  +
		relationTotal * m * 4 + relationTotal * dimension * 2, sizeof(REAL));
	matrix_h = entityVec + entityTotal * dimension;
	matrix_t = matrix_h + relationTotal * m;

	L_h=matrix_t+relationTotal*m;
	L_t=L_h+relationTotal*m;

	centr_h = L_t + relationTotal * m;
	centr_t = centr_h + relationTotal * dimension;

	A=(REAL *)calloc(relationTotal*dimension*dimension, sizeof(REAL));
	etmp=(REAL*)calloc(entityTotal*dimension,sizeof(REAL));
	
	
	for (INT i = 0; i < 2 * relationTotal; i++){
		INT last=i * m;
		for(INT ii = 0;ii < dimension; ii++)
			for(INT jj = ii; jj < dimension; jj++){
				if(ii==jj){
					matrix_h[last+ii*dimension+jj-ii*(ii+1)/2]=0.001+fabs(randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension)));
				}
				else
					matrix_h[last+ii*dimension+jj-ii*(ii+1)/2]=randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			}
	}
	for (INT i = 0; i < 2 * relationTotal; i++){
		for (INT ii=0; ii < dimension; ii++){
			centr_h[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		}
	}
	
	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainList = (Triple *)calloc(tripleTotal , sizeof(Triple));
	for (INT i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
	}
	fclose(fin);

	cout<<"inin done\n";
}

void load() {
	cout<<"load start\n";
	FILE *fin;
	INT tmp;
	fin = fopen((loadPath + "entity2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < entityTotal; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &etmp[last + j]);
	}
	fclose(fin);

	fin = fopen((loadPath + "A" + note + ".vec").c_str(), "r");
    for (long i = 0; i < relationTotal; i++)
            for (long jj = 0; jj < dimension; jj++)
                for (long ii = 0; ii < dimension; ii++)
                    tmp = fscanf(fin, "%f", &A[i * dimension * dimension + jj + ii * dimension]);
    fclose(fin);

    //memset(entityVec,0,entityTotal*dimension*sizeof(REAL));

  //  for(int i=0；i<entityTotal;i++){
   // 	int last=i*dimension;
   // 	for(int j=0;j<dimension;j++){
   // 		for(int k=0;k<dimension;k++)
	//    		entity2vec[last+j]+=A[]
    //	}
   // }
	cout<<"load done\n";
}

INT Len;
INT Batch;
REAL res;

void cal_M(INT rel,REAL *M_h,REAL *M_t){
	INT lastM = rel * dimension * (dimension+1)/2;
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++){
			for(INT k=0;k<=i;k++){
				M_h[i * dimension + j -  i*(i+1)/2] += matrix_h[lastM + k*dimension+i-k*(k+1)/2]*matrix_h[lastM + k*dimension+j-k*(k+1)/2];
				if(isnan(M_h[i * dimension + j -  i*(i+1)/2])) cerr<<"error";
			}
		}
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
			for(INT k=0;k<=i;k++){
				M_t[i * dimension + j -  i*(i+1)/2] += matrix_t[lastM + k*dimension+i-k*(k+1)/2]*matrix_t[lastM + k*dimension+j-k*(k+1)/2];
			}
}



REAL cal_k_h(INT e,INT rel, REAL *M){
	INT lasta = e * dimension;
	INT lastc = rel * dimension;
	REAL k=0;
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++){
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(entityVec[lasta+i]-centr_h[lastc+i])*(entityVec[lasta+j]-centr_h[lastc+j]);
		}
		
	return k;
}
REAL cal_k_t(INT e,INT rel, REAL *M){
	INT lasta = e * dimension;
	INT lastc = rel * dimension;
	REAL k=0;
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(entityVec[lasta+i]-centr_t[lastc+i])*(entityVec[lasta+j]-centr_t[lastc+j]);
	return k;
}

REAL cal_y_h(INT e,INT rel){
	INT lasta = e * dimension;
	INT lastc =  rel *dimension;
	REAL y=0;
	for(INT i=0;i<dimension;i++)
		y+=(entityVec[lasta+i]-centr_h[lastc+i])*(entityVec[lasta+i]-centr_h[lastc+i]);
	return y;
}
REAL cal_y_t(INT e,INT rel){
	INT lasta = e * dimension;
	INT lastc =  rel *dimension;
	REAL y=0;
	for(INT i=0;i<dimension;i++)
		y+=(entityVec[lasta+i]-centr_t[lastc+i])*(entityVec[lasta+i]-centr_t[lastc+i]);
	return y;
}

REAL calc_sum(INT eh, INT et, INT rel){
	INT lasta1 = eh * dimension;
	INT lasta2 = et * dimension;
	INT lastM = rel * dimension * (dimension+1)/2;
	INT lastc = rel * dimension;
	REAL M_h[dimension*(dimension+1)/2];
	REAL M_t[dimension*(dimension+1)/2];
	memset(M_h,0,dimension*(dimension+1)/2*sizeof(REAL));
	memset(M_t,0,dimension*(dimension+ 1)/2*sizeof(REAL));

	cal_M(rel, M_h,M_t);
	memset(entityVec+lasta1,0,dimension*sizeof(REAL));
	memset(entityVec+lasta2,0,dimension*sizeof(REAL));	
//for(int i=0；i<entityTotal;i++){
		//int last1=i*dimension;
		for(int j=0;j<dimension;j++){
			int last2=j*dimension;
			for(int k=0;k<dimension;k++){
				entityVec[lasta1+j]+=A[rel*dimension*dimension+last2+k]*etmp[lasta1+k];
				entityVec[lasta2+j]+=A[rel*dimension*dimension+last2+k]*etmp[lasta2+k];
			}
		}
	//}

	//memcpy(L_h+lastM,M_h,dimension*(dimension+1)/2);
	//memcpy(L_t+lastM,M_t,dimension*(dimension+1)/2);

	REAL k_h=cal_k_h(eh, rel, M_h);
	REAL k_t=cal_k_t(et, rel,M_t);
	REAL y_h=cal_y_h(eh,rel);
	REAL y_t=cal_y_t(et,rel);

	REAL d_h=pow(1-pow(k_h,-0.5),2)*y_h;
	REAL d_t=pow(1-pow(k_t,-0.5),2)*y_t;

	return d_h+d_t;
}

void gradient(INT eh, INT et, INT rel) {
	INT lasta1 = eh * dimension;
	INT lasta2 = et * dimension;
	INT lastM = rel * dimension * (dimension+1)/2;
	INT lastc = rel * dimension;

	REAL M_h[dimension*(dimension+1)/2];
	REAL M_t[dimension*(dimension+1)/2];

	memset(M_h,0,dimension*(dimension+1)/2*sizeof(REAL));
	memset(M_t,0,dimension*(dimension+ 1)/2*sizeof(REAL));

	cal_M(rel, M_h,M_t);

	memcpy(L_h+lastM,M_h,dimension*(dimension+1)/2*sizeof(REAL));
	memcpy(L_t+lastM,M_t,dimension*(dimension+1)/2*sizeof(REAL));

	REAL k_h=cal_k_h(eh, rel, M_h);
	REAL k_t=cal_k_t(et, rel,M_t);
	REAL y_h=cal_y_h(eh,rel);
	REAL y_t=cal_y_t(et,rel);

	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
		{	
			REAL z=0;
			for(INT k=i;k<dimension;k++)
				z+=2*(entityVec[lasta1+k]-centr_h[lastc+k])*(entityVec[lasta1+j]-centr_h[lastc+j])*matrix_h[lastM+i*dimension+k-i*(i+1)/2];
			matrix_h[lastM+i*dimension+j-i*(i+1)/2] -= alpha * y_h * (1-pow(k_h,-0.5)) *pow(k_h,-1.5)*z;
		}
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
		{	
			REAL z=0;
			for(INT k=i;k<dimension;k++)
				z+=2*(entityVec[lasta2+k]-centr_t[lastc+k])*(entityVec[lasta2+j]-centr_t[lastc+j])*matrix_t[lastM+i*dimension+k-i*(i+1)/2];
			matrix_t[lastM+i*dimension+j-i*(i+1)/2] -= alpha * y_t * (1-pow(k_t,-0.5)) *pow(k_t,-1.5)*z;
		}

	for(INT i=0;i<dimension;i++)
	{
		REAL z=0;
		for(INT k=0;k<dimension;k++){
			if(k<=i)
				z+=2*matrix_h[k*dimension+i-k*(k+1)/2]*(entityVec[lasta1+k]-centr_h[lastc+k]);
			else
				z+=2*matrix_h[i*dimension+k-i*(i+1)/2]*(entityVec[lasta1+k]-centr_h[lastc+k]);
		}
		centr_h[lastc+i]-=alpha*(pow(1-pow(k_h,-0.5),2)*2*(entityVec[lasta1+i]-centr_h[lastc+i])+y_h*(1-pow(k_h,-0.5)) *pow(k_h,-1.5)*z);
	}
	for(INT i=0;i<dimension;i++)
	{
		REAL z=0;
		for(INT k=0;k<dimension;k++){
			if(k<=i)
				z+=2*matrix_t[k*dimension+i-k*(k+1)/2]*(entityVec[lasta2+k]-centr_t[lastc+k]);
			else
				z+=2*matrix_t[i*dimension+k-i*(i+1)/2]*(entityVec[lasta2+k]-centr_t[lastc+k]);
		}
		centr_t[lastc+i]-=alpha*(pow(1-pow(k_t,-0.5),2)*2*(entityVec[lasta2+i]-centr_t[lastc+i])+y_t*(1-pow(k_t,-0.5)) *pow(k_t,-1.5)*z);
	}
}


void train_kb(INT e1_a, INT e2_a, INT rel_a) {
	REAL sum1 = calc_sum(e1_a, e2_a, rel_a);
	res+=sum1;
	gradient(e1_a, e2_a, rel_a);
}

void* trainMode(void *con) {
	INT id, pr, i, j;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (INT k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		train_kb(trainList[i].h, trainList[i].t, trainList[i].r);
	}
	pthread_exit(NULL);
}

void out() {
		FILE* f1 = fopen((outPath + "M" + note + ".vec").c_str(),"w");
		for (INT i = 0; i < 2 * relationTotal; i++)
			for (INT jj = 0; jj < dimension; jj++) {
				for (INT ii = jj; ii < dimension; ii++)
					fprintf(f1, "%f\t", L_h[i * dimension * (dimension+1)/2 + jj*dimension + ii -jj*(jj+1)/2]);
				fprintf(f1,"\n");
			}
		fclose(f1);
		FILE* f4 = fopen((outPath + "C" + note + ".vec").c_str(),"w");
		for (INT i = 0; i < relationTotal * 2; i++) {
			INT last = dimension * i;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f4, "%.6f\t", centr_h[last + ii]);
			fprintf(f4,"\n");
		}
		fclose(f4);
}

void* train(void *con) {
	Len = tripleTotal;
	Batch = Len / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
	for (INT epoch = 0; epoch < trainTimes; epoch++) {
		res = 0;
		for (INT batch = 0; batch < nbatches; batch++) {
			//res = 0;
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			//printf("batch %d %f\n", batch, res);
		}
		printf("epoch %d %f\n", epoch, res);

		if((epoch+1)%100==0)
			out();
	}

/*
	for(int i=0;i<relationTotal;i++){
		int lastM=i*dimension*(dimension+1)/2;
		for(INT i=0;i<dimension;i++)

		for(INT j=i;j<dimension;j++){
			for(INT k=0;k<=i;k++){
				L_h[lastM +i * dimension + j -  i*(i+1)/2] += matrix_h[lastM + k*dimension+i-k*(k+1)/2]*matrix_h[lastM + k*dimension+j-k*(k+1)/2];
				if(isnan(L_h[i * dimension + j -  i*(i+1)/2])) cerr<<"error";
			}
		}
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
			for(INT k=0;k<=i;k++){
				L_t[lastM +i * dimension + j -  i*(i+1)/2] += matrix_t[lastM + k*dimension+i-k*(k+1)/2]*matrix_t[lastM + k*dimension+j-k*(k+1)/2];
			}
	}
*/
}


int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void setparameters(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	//if ((i = ArgPos((char *)"-sizeR", argc, argv)) > 0) dimensionR = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outPath = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) loadPath = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) trainTimes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

int main(int argc, char **argv) {
	setparameters(argc, argv);
	init();
	if (loadPath != "") load();
	train(NULL);
	if (outPath != "") out();
	return 0;
}
