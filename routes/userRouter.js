import express from 'express';
import { adminLogin, loginUser, sendOtp, verifyOtp,registerUser, readImage, disposeWaste, addBin } from '../controllers/userController.js';
import upload from '../middleware/multer.js';



const userRouter=express.Router();

userRouter.post('/send-otp',sendOtp);
userRouter.post('/verify-otp',verifyOtp);
userRouter.post('/register',registerUser);
userRouter.post('/login',loginUser);
userRouter.post('/admin',adminLogin);
userRouter.post('/image', upload.single('image'), readImage);
userRouter.post('/dispose',disposeWaste)
userRouter.post('/bin',addBin)
userRouter.get('/check',checkMe)

export default userRouter;