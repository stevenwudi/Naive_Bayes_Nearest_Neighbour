% This is am implemenation of 
% " EigenJoints-based Actioin Recognition Using
% Naive-Bayes-Nearest-Neighbor"
%  by Di Wu-- stevenwudi@gmail.com 2012-11-11 (¹â¹÷½Ú£©haha

clc;clear;close all;

training=1;
if training
%% Feature extraction
FeaturesAll=[];
load('ActionTrain');
action=1;
    for i=1:length(ActionTrain)
        PoseAll=ActionTrain(i).poseData;
        FeaturesPoolTemp=[];
        for j=1:length(ActionTrain(i).ActionData)
            action=action+1;
            Pose=PoseAll(ActionTrain(i).ActionData(j).marg_ind(1):ActionTrain(i).ActionData(j).marg_ind(end),:,:);
            clear Fcc Fcp Fci
            % f_cc
            FeatureNum=1;
            for joints1=1:19
                for joints2=joints1+1:20
                    Fcc(FeatureNum,:,:)=Pose(:,joints1,:)-Pose(:,joints2,:);
                    FeatureNum=FeatureNum+1;
                end
            end
            
            % f_cp
            FeatureNum=1;
            for joints1=1:20
                for joints2=1:20
                        Fcp(FeatureNum ,:,:)=Pose(2:end,joints1,:)-Pose(1:end-1,joints2,:);
                        FeatureNum=FeatureNum+1;
                end
           end
                
            %f_ci
            FeatureNum=1;
            Pose_init=repmat(Pose(1,:,:),size(Pose,1)-1,1);
            for joints1=1:20
                for joints2=1:20
                        Fci(FeatureNum ,:,:)=Pose(2:end,joints1,:)-Pose_init(:,joints2,:);
                        FeatureNum=FeatureNum+1;
                end
            end
             
                Fcc=permute(Fcc,[2 1 3]);
                Fcp=permute(Fcp,[2 1 3]);
                Fci=permute(Fci,[2 1 3]);
                
                Features=FlatenPoses(cat(2,Fcc(1:end-1,:,:), Fcp , Fci));
            
                seqs{action}=Features;   
                FeaturesAll=[FeaturesAll;Features];
                FeaturesPoolTemp=[FeaturesPoolTemp;Features];
        end
        FeaturePool{i}=FeaturesPoolTemp;
    end
    
    %% PCA dimensionality reduction
    
    optionsPCA.ReducedDim=128;
    %optionsPCA.PCARatio=0.95;
   % [eigvector, eigvalue, meanData, new_data] = PCA(FeaturesAll, optionsPCA);
    save('FeaturesAll','FeaturesAll');
    save('FeaturePool','FeaturePool');
    save('eigvector','eigvector');
end   
   
clc;clear;close all;

%% Memory based testing
load('ActionTest');
load('FeaturePool');
load('eigvector');
%% Feature extraction
FeaturesAll=[];
action=0;
AS1CrSub=[2 3 5 6 10 13 18 20];
AS2CrSub=[1 4 7 8 9 11 14 12];
AS3CrSub=[6,14:20];
TestSequence=AS3CrSub;
%TestSequence=1:20;
    for i=TestSequence
        PoseAll=ActionTest(i).poseData;
        FeaturesPoolTemp=[];
        for j=1:length(ActionTest(i).ActionData)
            action=action+1;
            Pose=PoseAll(ActionTest(i).ActionData(j).marg_ind(1):ActionTest(i).ActionData(j).marg_ind(end),:,:);
            clear Fcc Fcp Fci
            % f_cc
            FeatureNum=1;
            for joints1=1:19
                for joints2=joints1+1:20
                    Fcc(FeatureNum,:,:)=Pose(:,joints1,:)-Pose(:,joints2,:);
                    FeatureNum=FeatureNum+1;
                end
            end
            
            % f_cp
            FeatureNum=1;
            for joints1=1:20
                for joints2=1:20
                        Fcp(FeatureNum ,:,:)=Pose(2:end,joints1,:)-Pose(1:end-1,joints2,:);
                        FeatureNum=FeatureNum+1;
                end
           end
                
            %f_ci
            FeatureNum=1;
            Pose_init=repmat(Pose(1,:,:),size(Pose,1)-1,1);
            for joints1=1:20
                for joints2=1:20
                        Fci(FeatureNum ,:,:)=Pose(2:end,joints1,:)-Pose_init(:,joints2,:);
                        FeatureNum=FeatureNum+1;
                end
            end
             
                Fcc=permute(Fcc,[2 1 3]);
                Fcp=permute(Fcp,[2 1 3]);
                Fci=permute(Fci,[2 1 3]);
                
                Features=FlatenPoses(cat(2,Fcc(1:end-1,:,:), Fcp , Fci));

                % NBNN
                clear Distance;

                for classNum=TestSequence
                        X=Features*eigvector;
                        Y=FeaturePool{classNum}*eigvector;
                        [d] = cvEucdist(X', Y');
                        Distance(classNum)=sum(min(d,[],2));
                end
                Distance(Distance==0)=inf;
                [~,ClassPredict(action)]=min(Distance);
                ClassTrue(action)=i;
                ClassMax(i,j)=ClassPredict(action);
                fprintf('Predict: %d, True: %d\n',ClassPredict(action),i);
        end
    end
    
    Accuracy=sum(ClassPredict==ClassTrue)/length(ClassTrue);
    fprintf('NBNN accuracy for 20 classes is :%f\n',Accuracy);
   
    %NBNN accuracy for 20 classes is :0.723906

    
  %AS1CrSub  NBNN accuracy for 20 classes is :0.764706
  %AS2CrSub  NBNN accuracy for 20 classes is :0.663866
  %AS3CrSub  NBNN accuracy for 20 classes is :0.915966