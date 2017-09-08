/*�������ݿⲢĬ�����Ժ������ʹ�øÿ�*/
libname a "d:\xu.liu\����\0906";
options compress=yes user=a;
/*�����ļ���������ʽ���ܴ��ڼ����Ե����⣬�Ƽ�csv*/
PROC IMPORT DATAFILE='d:\xu.liu\����\rm0906.csv' 
OUT=work DBMS=CSV REPLACE;
GUESSINGROWS=2000;
RUN;

/*���������ֶ�*/
data aim;
set e(keep=MEMBER_CODE trade_id signphone_prov signphone_city bankphone_prov bankphone_city idcard_prov idcard_city bankid_city bankid_prov kyh_sign_prov kyh_sign_city);
run;

/*��һ������ʡ������*/
/*����ʡ�ݼ���*/
data e;
set Aim_analyst_111610;
run;
/*�����ָ�ʽ��Ϊ�ַ�����ʽ����ȥ�ո�*/
data work_1;
set work_1;
member_code_c=input(member_code,$20.);
member_code_c=COMPRESS(member_code);
run;
/*��һ�����ݼ���ʽ*/
proc contents data=work_1;quit;
/*����sql�ӿڽ�����ϲ��������±�*/
proc sql;
    create table enew as
    select e.*,b.province as b_province, b.city as b_city
    from e 
	left join work_1 as b
    on e.member_code=b.member_code_c;
quit;
/*ͳ��ĳ�ֶ�*/
proc freq data=enew noprint;
tables b_province/missing nocol norow nopercent out=e_prov6;
run;
/*�꣺�鿴�ֶ�var�У���Ϊvar1���滻Ϊvar2*/
%macro check(var,var1,var2);
data enew;
set enew;
&var.=compress(&var.);
if &var.=&var1. then &var.=&var2.;
run;
%mend check;
%check(b_province,'����׳��','����');%check(b_province,'�½�ά���','�½�');
/*�����е���Դ��ͬ��ʡ�ݺ�Ϊһ��*/
data prov;
length prov $16.;
set e_prov1(keep=signphone_prov rename=(signphone_prov=prov)) e_prov2(keep=bankphone_prov rename=(bankphone_prov=prov))
e_prov3(keep=idcard_prov rename=(idcard_prov=prov)) e_prov4(keep=bankid_prov rename=(bankid_prov=prov)) 
e_prov5(keep=kyh_sign_prov rename=(kyh_sign_prov=prov)) e_prov6(keep=b_province rename=(b_province=prov));
run;
/*����ʡ������*/
proc sort data=prov nodupkey;
by prov;
run;
/*%macro check(var,var1,var2);*/
/*data e;*/
/*set e;*/
/*&var.=compress(&var.);*/
/*if &var.=&var1. then &var.=&var2.;*/
/*run;*/
/*%mend check;*/


/*���ɳ��м���*/
proc freq data=enew noprint;
tables signphone_city/missing nocol norow nopercent out=city1;
run;
proc freq data=enew noprint;
tables bankphone_city/missing nocol norow nopercent out=city2;
run;
proc freq data=enew noprint;
tables idcard_city/missing nocol norow nopercent out=city3;
run;
proc freq data=enew noprint;
tables bankid_city/missing nocol norow nopercent out=city4;
run;
proc freq data=enew noprint;
tables kyh_sign_city/missing nocol norow nopercent out=city5;
run;
proc freq data=enew noprint;
tables b_city/missing nocol norow nopercent out=city6;
run;

/*��������޸��б�*/
PROC IMPORT DATAFILE='D:\rm\0906\city_dic.csv' 
OUT=city_dic_1 DBMS=CSV REPLACE;
GUESSINGROWS=20;
RUN;

/*�꣬�����滻�ϲ�*/
%macro a;
%let dsid=%sysfunc(open(a.city_dic_1,i));/*city-dic_1Ϊ�ֵ��*/
%let num=%sysfunc(attrn(&dsid,nobs));
   %do seq=1 %to %eval(&num.);
     %let rc=%sysfunc(fetchobs(&dsid,&seq));
     %let original=%sysfunc(getvarc(&dsid,%sysfunc(varnum(&dsid,original))));
     %let modfied=%sysfunc(getvarc(&dsid,%sysfunc(varnum(&dsid,modfied))));
/*	 �ֶ�signphone_city��in��city_dic_1��original�ֶ��У����½�modified�и���*/
		 if signphone_city="&original." then modified="&modfied.";
	%end;
%mend;

/*�޸����������Ϣ*/

DATA a.Enew_A;
SET a.Enew;
%a;
RUN;

data city;
length city $30.;
set city1(keep=signphone_city rename=(signphone_city=city)) city2(keep=bankphone_city rename=(bankphone_city=city))
city3(keep=idcard_city rename=(idcard_city=city)) city4(keep=bankid_city rename=(bankid_city=city))
city5(keep=kyh_sign_city rename=(kyh_sign_city=city)) city6(keep=b_city rename=(b_city=city));
run;
/*��һ��*/
proc sort data=city nodupkey;
by city;
run;
/*�ļ�����*/
proc export data=enew_a
outfile="d:\xu.liu\����\0906.csv"
DBMS=csv  REPLACE;
run;
