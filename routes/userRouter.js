import express from 'express';
import { adminLogin, loginUser, sendOtp, verifyOtp,registerUser, readImage, disposeWaste, addBin, checkMe, submitWasteInfo } from '../controllers/userController.js';
import upload from '../middleware/multer.js';
import { userAuth } from '../middleware/userAuth.js';



const userRouter=express.Router();

userRouter.post('/send-otp',sendOtp);
userRouter.post('/verify-otp',verifyOtp);
userRouter.post('/register',registerUser);
userRouter.post('/login',loginUser);
userRouter.post('/admin',adminLogin);
userRouter.post('/image', userAuth,upload.single('image'), readImage);
userRouter.post('/dispose',disposeWaste)
userRouter.post('/bin',addBin)
userRouter.get('/check',checkMe)
userRouter.post('/submitwaste',submitWasteInfo)

export default userRouter;